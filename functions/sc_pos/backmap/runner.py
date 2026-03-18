from __future__ import annotations

"""sc_pos.backmap.runner

Single-entry, dict-driven execution for backmapping + PFSS context + optional movies.

The runner exists to keep notebooks/MREs simple:
    out = run_pfss_backmap_pipeline(cfg)

where cfg is a single dictionary typically built as the deep merge of several
category dictionaries (paths/run/backmap/plots/pfss/movies).
"""

from dataclasses import is_dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple


def _make_dataclass(cls: Any, kwargs: Mapping[str, Any]) -> Any:
    """Instantiate a dataclass-like type, ignoring unsupported kwargs."""
    import inspect

    try:
        sig = inspect.signature(cls)
        allowed = set(sig.parameters.keys())
        filt = {k: v for k, v in dict(kwargs).items() if k in allowed}
        return cls(**filt)
    except Exception:
        # Fall back: try direct construction
        return cls(**dict(kwargs))


def run_pfss_backmap_pipeline(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    """Run backmapping (+ optional PFSS products and movies) from a single config dict."""

    # Local imports keep heavy deps out of import-time paths.
    from .pipeline import run_backmapping_interval, run_backmapping_range
    from .movie import MovieConfig, Movie3DConfig, make_pfss_backmap_movie, make_pfss_backmap_movie_3d

    cfg0: Dict[str, Any] = dict(cfg)

    root_dir = Path(cfg0.get("root_dir"))
    sc = str(cfg0.get("sc"))
    outdir = cfg0.get("outdir", None)

    # -------------------------
    # Backmapping core settings
    # -------------------------
    method = str(cfg0.get("method", "ballistic_bg"))
    cadence = str(cfg0.get("cadence", "60min"))
    ephem_step = str(cfg0.get("ephem_step", "1h"))
    r_ss_rsun = float(cfg0.get("r_ss_rsun", 2.5))
    omega_deg_per_day = float(cfg0.get("omega_deg_per_day", 14.1844))
    phi_sign = int(cfg0.get("phi_sign", +1))
    remove_gaps = bool(cfg0.get("remove_gaps", True))
    gap_pad_frac = float(cfg0.get("gap_pad_frac", 0.5))
    model_kwargs = cfg0.get("model_kwargs", None)

    # -------------------------
    # Plot settings
    # -------------------------
    plot_vars = cfg0.get("plot_vars", ("polarity", "Vr_bg", "P_ram", "sigma_c"))
    plot_3d = bool(cfg0.get("plot_3d", False))
    plot_3d_vars = cfg0.get("plot_3d_vars", None)
    plot_3d_ncols = int(cfg0.get("plot_3d_ncols", 2))
    plot_3d_decimate = int(cfg0.get("plot_3d_decimate", 1))
    plot_carrington_diag = bool(cfg0.get("plot_carrington_diag", True))
    frame3d = str(cfg0.get("frame3d", "HEE"))
    show = bool(cfg0.get("show", False))
    verbose = bool(cfg0.get("verbose", True))

    # -------------------------
    # PFSS settings
    # -------------------------
    pfss_cfg = dict(cfg0.get("pfss", {}) or {})
    pfss_enabled = bool(pfss_cfg.get("enabled", False))
    pfss_config_for_pipeline: Optional[Dict[str, Any]] = None
    if pfss_enabled:
        # The pipeline expects a flat dict; we pass through keys used by pfss.py
        # and pipeline.py. Unknown keys are ignored by the pipeline.
        pfss_config_for_pipeline = dict(pfss_cfg)

    # -------------------------
    # Run selection
    # -------------------------
    run_mode = str(cfg0.get("run_mode", "single")).strip().lower()
    if run_mode not in {"single", "range"}:
        raise ValueError("cfg['run_mode'] must be 'single' or 'range'")

    if run_mode == "single":
        which_int = int(cfg0.get("which_int", 0))
        res = run_backmapping_interval(
            root_dir=root_dir,
            sc=sc,
            which_int=which_int,
            method=method,
            cadence=cadence,
            ephem_step=ephem_step,
            r_ss_rsun=r_ss_rsun,
            omega_deg_per_day=omega_deg_per_day,
            phi_sign=phi_sign,
            remove_gaps=remove_gaps,
            gap_pad_frac=gap_pad_frac,
            model_kwargs=model_kwargs,
            plot_vars=plot_vars,
            plot_3d=plot_3d,
            plot_3d_vars=plot_3d_vars,
            plot_3d_ncols=plot_3d_ncols,
            plot_3d_decimate=plot_3d_decimate,
            plot_carrington_diag=plot_carrington_diag,
            frame3d=frame3d,
            pfss_config=pfss_config_for_pipeline,
            outdir=outdir,
            show=show,
            verbose=verbose,
        )
        out_main: Dict[str, Any] = dict(res)

    else:
        start = str(cfg0.get("start"))
        end = str(cfg0.get("end"))
        selection = str(cfg0.get("selection", "start_in_range"))
        merge_intervals = bool(cfg0.get("merge_intervals", True))
        save_per_interval = bool(cfg0.get("save_per_interval", False))

        res = run_backmapping_range(
            root_dir=root_dir,
            sc=sc,
            start=start,
            end=end,
            selection=selection,
            method=method,
            cadence=cadence,
            outdir=outdir,
            merge_intervals=merge_intervals,
            save_per_interval=save_per_interval,
            # kwargs forwarded into run_backmapping_interval
            ephem_step=ephem_step,
            r_ss_rsun=r_ss_rsun,
            omega_deg_per_day=omega_deg_per_day,
            phi_sign=phi_sign,
            remove_gaps=remove_gaps,
            gap_pad_frac=gap_pad_frac,
            model_kwargs=model_kwargs,
            plot_vars=plot_vars,
            plot_3d=plot_3d,
            plot_3d_vars=plot_3d_vars,
            plot_3d_ncols=plot_3d_ncols,
            plot_3d_decimate=plot_3d_decimate,
            plot_carrington_diag=plot_carrington_diag,
            frame3d=frame3d,
            pfss_config=pfss_config_for_pipeline,
            show=show,
            verbose=verbose,
        )

        # Prefer merged output if present.
        merged = res.get("merged", None)
        if isinstance(merged, dict) and isinstance(merged.get("files", None), dict):
            out_main = dict(merged)
        else:
            out_main = dict(res)

    # -------------------------
    # Movies (optional)
    # -------------------------
    movies = dict(cfg0.get("movies", {}) or {})
    made: Dict[str, str] = {}
    if bool(movies.get("enabled", False)):
        if not pfss_enabled:
            raise ValueError("movies.enabled=True requires pfss.enabled=True (movies need PFSS maps).")

        files = out_main.get("files", {}) if isinstance(out_main.get("files", None), dict) else {}
        ts_path = files.get("timeseries", None)
        pfss_outdir = files.get("pfss_outdir", None)
        out_base = files.get("outdir", None)
        if (not ts_path) or (not pfss_outdir) or (not out_base):
            raise RuntimeError("Missing timeseries/pfss_outdir/outdir from pipeline output; cannot build movies.")

        out_base = Path(str(out_base))

        # PFSS numeric settings for map loading
        pfss_N = int(pfss_cfg.get("N", 180))
        pfss_rss = float(pfss_cfg.get("rss_rsun", r_ss_rsun))
        pfss_nr = pfss_cfg.get("nr", None)
        pfss_prefer = str(pfss_cfg.get("prefer_hhmm", "1204"))
        pfss_overwrite = bool(pfss_cfg.get("overwrite_download", False))
        pfss_cache = bool(pfss_cfg.get("cache_maps", True))

        # Optional fixed clim shared across movies
        clim: Optional[Tuple[float, float]] = None
        clim_mode = str(pfss_cfg.get("clim_mode", "global")).strip().lower()
        if clim_mode == "fixed":
            cf = pfss_cfg.get("clim_fixed", None)
            if (isinstance(cf, (list, tuple)) and len(cf) == 2):
                clim = (float(cf[0]), float(cf[1]))

        # 2D MP4
        if bool(movies.get("make_2d", False)):
            out_mp4 = out_base / str(movies.get("name_2d", "pfss_backmap_merged.mp4"))
            cfg2d = _make_dataclass(
                MovieConfig,
                dict(
                    out_mp4=out_mp4,
                    which_br=str(movies.get("which_br", pfss_cfg.get("visual_br", pfss_cfg.get("which_br", "source_surface")))),
                    fps=int(movies.get("fps_2d", 12)),
                    stride=int(max(1, int(movies.get("stride", 1)))),
                    tail=int(max(0, int(movies.get("tail", 240)))),
                    panel_vars=tuple(movies.get("panel_vars_2d", movies.get("panel_vars", ("sigma_c", "sigma_r")))),
                    title=str(movies.get("title", "Backmapping + PFSS")),
                    dpi=int(movies.get("dpi", 150)),
                    mode=str(movies.get("mode", "reuse")),
                    n_jobs=int(movies.get("n_jobs", -1)),
                    overwrite=True,
                    pfss_date_mode=str(movies.get("pfss_date_mode", "from_data")),
                    pfss_date_str=str(movies.get("pfss_date_str", "")),
                ),
            )

            mp4 = make_pfss_backmap_movie(
                data=Path(str(ts_path)),
                cfg=cfg2d,
                pfss_out_dir=Path(str(pfss_outdir)),
                N=pfss_N,
                rss_rsun=pfss_rss,
                nr=(int(pfss_nr) if pfss_nr is not None else None),
                prefer_hhmm=pfss_prefer,
                overwrite_download=pfss_overwrite,
                clim=clim,
            )
            made["movie_2d"] = str(mp4)
        # 3D MP4 (Matplotlib)
        if bool(movies.get("make_3d", False)):
            out_mp4 = out_base / str(movies.get("name_3d", "pfss_backmap_3d.mp4"))

            # Prefer stride_3d; if unset, use 0 (auto-selection inside the 3D renderer).
            stride3d = int(movies.get("stride_3d", 0))

            # Base kwargs (legacy keys kept stable)
            cfg3d_kwargs: Dict[str, Any] = dict(
                out_mp4=out_mp4,
                which_br=str(movies.get("which_br", "photosphere")),
                fps=int(movies.get("fps_3d", 3)),
                stride=int(stride3d),
                tail=int(max(0, int(movies.get("tail", 240)))),
                panel_vars=tuple(movies.get("panel_vars_3d", movies.get("panel_vars", ()))),
                title=str(movies.get("title_3d", movies.get("title", "PFSS (photosphere) + source-surface backmapping + spacecraft trajectory"))),
                dpi=int(movies.get("dpi_3d", movies.get("dpi", 150))),
                overwrite=True,
                pfss_date_mode=str(movies.get("pfss_date_mode", "from_data")),
                pfss_date_str=str(movies.get("pfss_date_str", "")),
            )

            # Pass-through extra Movie3DConfig keys (safe: _make_dataclass filters unsupported)
            extras = dict(movies)
            for kk in (
                "enabled", "make_2d", "make_3d",
                "name_2d", "name_3d",
                "fps_2d", "fps_3d",
                "stride", "stride_3d",
                "mode", "n_jobs",
                "pfss_date_mode", "pfss_date_str",
                "which_br", "tail", "title", "title_3d",
                "dpi", "dpi_3d",
            ):
                extras.pop(kk, None)
            cfg3d_kwargs.update(extras)

            cfg3d = _make_dataclass(Movie3DConfig, cfg3d_kwargs)

            mp4 = make_pfss_backmap_movie_3d(
                data=Path(str(ts_path)),
                cfg=cfg3d,
                pfss_out_dir=Path(str(pfss_outdir)),
                N=pfss_N,
                rss_rsun=pfss_rss,
                nr=(int(pfss_nr) if pfss_nr is not None else None),
                prefer_hhmm=pfss_prefer,
                overwrite_download=pfss_overwrite,
                cache_maps=pfss_cache,
                clim=clim,
            )
            made["movie_3d"] = str(mp4)

    if made:
        out_main["movies"] = made

    return out_main
