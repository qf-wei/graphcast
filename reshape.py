#!/usr/bin/env python
import datetime as dt
import os
from pathlib import Path
import numpy as np
import xarray as xr
from graphcast import graphcast

AR_STORE = (
    "gs://gcp-public-data-arco-era5/ar/"
    "1959-2022-1h-360x181_equiangular_with_poles_conservative.zarr"
)

TARGET_ROOT = Path("generated_graphcast_inputs")

PRESSURE_VARS = [
    "u_component_of_wind", "v_component_of_wind", "temperature",
    "specific_humidity", "vertical_velocity", "geopotential",
]

SURFACE_VARS = [
    "2m_temperature", "sea_surface_temperature", "mean_sea_level_pressure",
    "10m_u_component_of_wind", "10m_v_component_of_wind", "total_precipitation",
]

STATIC_VARS = [
    "land_sea_mask", "geopotential_at_surface",
]

PRESSURE_LEVELS_13 = list(graphcast.PRESSURE_LEVELS_WEATHERBENCH_13)
ALL_VARS = STATIC_VARS + SURFACE_VARS + PRESSURE_VARS

def add_time_features(ds: xr.Dataset) -> xr.Dataset:
    # ⚠️ 元の時間（datetime64）を一時的に使うために datetime を使う
    # 既に datetime64 な座標があればそれを使い、なければ relative_time を使う
    if "datetime" in ds.coords:
        time_dt = xr.DataArray(ds["datetime"].data, coords={"time": ds.time}, dims=("time",))
    else:
        raise ValueError("datetime coordinate is required for time feature generation.")

    secs = (time_dt.dt.hour * 3600 +
            time_dt.dt.minute * 60 +
            time_dt.dt.second)
    day_angle = 2 * np.pi * secs / 86400

    year_len = xr.where(
        ((time_dt.dt.year % 4 == 0) & ((time_dt.dt.year % 100 != 0) | (time_dt.dt.year % 400 == 0))),
        366, 365,
    )
    year_angle = 2 * np.pi * (time_dt.dt.dayofyear - 1) / year_len

    # ① 時間だけの DataArray を作成
    dc = xr.DataArray(np.cos(day_angle).data, coords={"time": ds.time}, dims=("time",))
    dsin = xr.DataArray(np.sin(day_angle).data, coords={"time": ds.time}, dims=("time",))

    # ② lon 方向へ broadcast → (time, lon)
    lon_name = "longitude" if "longitude" in ds.coords else "lon"
    dc   = dc.expand_dims({lon_name: ds[lon_name]}, axis=1)
    dsin = dsin.expand_dims({lon_name: ds[lon_name]}, axis=1)

    # ③ 年進行は (time,) のまま
    yc = xr.DataArray(np.cos(year_angle).data, coords={"time": ds.time}, dims=("time",))
    ys = xr.DataArray(np.sin(year_angle).data, coords={"time": ds.time}, dims=("time",))

    return ds.assign(
        day_progress_cos=dc,
        day_progress_sin=dsin,
        year_progress_cos=yc,
        year_progress_sin=ys,
    )

class ERA5ToGraphCast:
    def __init__(self, date: dt.date, lead_hours: int = 6,
                 token_path: str | None = None, out_root: Path = TARGET_ROOT):
        self.date = date
        self.lead_h = lead_hours
        self.token = token_path or os.path.expanduser(
            "~/.config/gcloud/application_default_credentials.json"
        )
        self.out_root = out_root

    def _open(self) -> xr.Dataset:
        return xr.open_zarr(
            AR_STORE,
            consolidated=True,
            chunks={"time": 1},
            storage_options={"token": self.token},
        )

    def _process(self, ds: xr.Dataset) -> xr.Dataset:
        # ---------- 3 つの時刻を切り出す ---------------------------------
        t0 = np.datetime64(f"{self.date}T00:00")
        t1 = t0 + np.timedelta64(self.lead_h, "h")
        t2 = t0 + np.timedelta64(self.lead_h * 2, "h")

        ds = ds.sel(time=slice(t0 - np.timedelta64(11, "h"), t2))
        available_vars = [v for v in ALL_VARS if v in ds.data_vars]
        ds = ds.sel(level=PRESSURE_LEVELS_13)[available_vars]

        # ---------- 12 h 積算降水量 --------------------------------------
        tp12 = (
            ds["total_precipitation"]
            .rolling(time=12, min_periods=12)
            .sum()
            .rename("total_precipitation_6hr")
        )
        ds = ds.assign(total_precipitation_6hr=tp12).drop_vars("total_precipitation")

        # ---------- t0,t1,t2 だけ残す ------------------------------------
        wanted = [t0, t1, t2]
        actual = np.array(ds.time.values, dtype="datetime64[ns]")
        sel = [actual[np.abs(actual - w).argmin()] for w in wanted]
        ds = ds.sel(time=sel)

        # ✅ 相対時間に変換（timedelta64[ns] 型に直す）
        relative_time = ds.time.data - t0
        ds = ds.assign_coords(time=relative_time)
        ds = ds.assign_coords(datetime=("time", sel))

        # ---------- 時間特徴量 ------------------------------------------
        ds = add_time_features(ds)

        # ---------- time を持つ変数に batch 次元を追加 -------------------
        new_vars = {}
        for v in ds.data_vars:
            if "time" in ds[v].dims:
                new_vars[v] = ds[v].expand_dims("batch", axis=0)
            else:
                new_vars[v] = ds[v]

        ds_out = xr.Dataset(new_vars, coords=ds.coords)

        # ---------- lat/lon 名にそろえ、並び替え -------------------------
        ds_out = ds_out.rename({"latitude": "lat", "longitude": "lon"})
        ds_out = ds_out.transpose("batch", "time", "level", "lat", "lon", missing_dims="ignore")

        # ---------- 検証が要求する追加座標 -------------------------------
        ds_out = ds_out.assign_coords(
            datetime=("time", np.array([t0, t1, t2])),  # ← 絶対時間を補助座標に入れる
            lat=("lat", ds_out.lat.data),
            lon=("lon", ds_out.lon.data),
        )

        return ds_out


    def build(self) -> Path:
        ds = self._process(self._open())
        out_name = (
            f"source-era5_date-{self.date:%Y-%m-%d}"
            f"_res-1.0_levels-{len(PRESSURE_LEVELS_13):02d}"
            f"_steps-{self.lead_h//6:02d}.nc"
        )
        out_path = self.out_root / out_name
        out_path.parent.mkdir(parents=True, exist_ok=True)
        enc = {v: {"zlib": True, "complevel": 4} for v in ds.data_vars}
        ds.to_netcdf(out_path, format="NETCDF4", encoding=enc)
        print("✅ saved:", out_path)
        return out_path

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser("ARCO ERA5 ➜ GraphCast NetCDF")
    p.add_argument("--date", default="2019-03-29", help="init date YYYY-MM-DD (00 UTC)")
    p.add_argument("--hours", type=int, default=12, help="forecast lead (multiple of 6 h)")
    args = p.parse_args()

    ERA5ToGraphCast(
        date=dt.datetime.strptime(args.date, "%Y-%m-%d").date(),
        lead_hours=args.hours,
    ).build()
