from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
from matplotlib import pyplot as plt
import scipy.stats as stats
from sklearn.neighbors import KernelDensity

from .. import logging as logg
from .._compat import old_positionals
from .._settings import settings
from .._utils import _doc_params
from ._baseplot_class import BasePlot, doc_common_groupby_plot_args
from ._docs import doc_common_plot_args, doc_show_save_ax, doc_vboundnorm
from ._utils import (
    ColorLike,
    _AxesSubplot,
    check_colornorm,
    fix_kwds,
    make_grid_spec,
    savefig_or_show,
)

if TYPE_CHECKING:
    from collections.abc import (
        Mapping,  # Special
        Sequence,  # ABCs
    )

    import pandas as pd
    from anndata import AnnData
    from matplotlib.axes import Axes
    from matplotlib.colors import Normalize
    from ._baseplot_class import _VarNames

class RidgePlot(BasePlot):

    DEFAULT_SAVE_PREFIX = "ridgeplot_"
    DEFAULT_LEGENDS_WIDTH = 0
    DEFAULT_CATEGORY_WIDTH = 10
    DEFAULT_COLORMAP = "gist_ncar"

    def __init__(
        self,
        adata: AnnData,
        var_names: _VarNames | Mapping[str, _VarNames],
        groupby: str | Sequence[str],
        *,
        bandwidth: float,
        palette,
        use_raw: bool | None = None,
        log: bool = False,
        num_categories: int = 7,
        categories_order: Sequence[str] | None = None,
        title: str | None = None,
        figsize: tuple[float, float] | None = None,
        gene_symbols: str | None = None,
        var_group_positions: Sequence[tuple[int, int]] | None = None,
        var_group_labels: Sequence[str] | None = None,
        var_group_rotation: float | None = None,
        layer: str | None = None,
        expression_cutoff: float = 0.0,
        mean_only_expressed: bool = False,
        standard_scale: Literal["var", "group"] | None = None,
        ridge_df: pd.DataFrame | None = None,
        ax: _AxesSubplot | None = None,
        vmin: float | None = None,
        vmax: float | None = None,
        vcenter: float | None = None,
        norm: Normalize | None = None,
        **kwds,
        ):

        BasePlot.__init__(
            self,
            adata,
            var_names,
            groupby,
            use_raw=use_raw,
            log=log,
            num_categories=num_categories,
            categories_order=categories_order,
            title=title,
            figsize=figsize,
            gene_symbols=gene_symbols,
            var_group_positions=var_group_positions,
            var_group_labels=var_group_labels,
            var_group_rotation=var_group_rotation,
            layer=layer,
            ax=ax,
            vmin=vmin,
            vmax=vmax,
            vcenter=vcenter,
            norm=norm,
            **kwds,
        )


        if ridge_df is None:
            self.ridge_df = self.obs_tidy
            self.group_indices = self.obs_tidy.groupby(level = 0, observed = True).indices
        else:
            self.ridge_df = ridge_df

        if palette is None:
            colormap = plt.get_cmap(self.DEFAULT_COLORMAP)
            self.palette = colormap(np.linspace(0, 1, len(self.group_indices)))
        else:
            self.palette = palette
        self.groupby = groupby
        self.bandwidth = bandwidth

    def _mainplot(self, ax):

        normalize = self._ridgeplot(ridge_ax = ax,
                                    ridge_df = self.ridge_df,
                                    ridge_group_indices = self.group_indices,
                                    ridge_groupby = self.groupby,
                                    bandwidth = self.bandwidth,
                                    palette = self.palette)

        return normalize

    @staticmethod
    def _ridgeplot(
            ridge_ax,
            ridge_df,
            ridge_group_indices,
            ridge_groupby,
            *,
            bandwidth,
            palette,
            color_on: str | None = "dot",
            y_label: str | None = None,
            dot_max: float | None = None,
            dot_min: float | None = None,
            standard_scale: Literal["var", "group"] = None,
            size_exponent: float | None = 2,
            edge_color: ColorLike | None = None,
            edge_lw: float | None = None,
            grid: bool | None = False,
            x_padding: float | None = 0.8,
            y_padding: float | None = 1.0,
            vmin: float | None = None,
            vmax: float | None = None,
            vcenter: float | None = None,
            norm: Normalize | None = None,
            **kwds,
            ):

            tracks = ridge_group_indices.keys()
            ntracks = len(tracks)
            exprarr = ridge_df.iloc[:,0].to_numpy()
            xmax = exprarr.max()
            head_crop = np.quantile(exprarr, 0.999)
            ridge_ax.set_prop_cycle('color', palette)
            ridge_ax.set_xlabel("Expression")
            ridge_ax.set_xlim((0,head_crop))
            ridge_ax.set_ylabel(ridge_groupby)
            ridge_ax.set_ylim((0,ntracks+1))
            ridge_ax.set_yticks(ticks = np.arange(0,len(tracks),1)+0.5, labels =
                                tracks)
            for idx,group in enumerate(tracks):
                x = ridge_df.iloc[ridge_group_indices[group],0].to_numpy().reshape(-1,1)
                kde = KernelDensity(kernel = "gaussian", bandwidth = bandwidth)
                kde.fit(x)
                xx = np.arange(0, xmax, 0.01).reshape(-1,1)
                k = kde.score_samples(xx)
                plot_y = np.exp(k)+idx
                plot_x = xx.flatten()
                z1=((ntracks*2)-(idx*2))
                z2=((ntracks*2)-((idx*2)-1))
                ridge_ax.plot(plot_x, plot_y, zorder=z1, color="black")
                ridge_ax.fill_between(x=plot_x, y1=plot_y, y2=[idx]*plot_y.shape[0], alpha = 1, zorder=z2)

            return check_colornorm(None, None, None, None)

    @staticmethod
    def has_zero_range(x: np.typing.ArrayLike):
        if x.min() == x.max():
            return True
        else:
            return False

def ridgeplot(
    adata: AnnData,
    var_names: _VarNames | Mapping[str, _VarNames],
    groupby: str | Sequence[str],
    *,
    bandwidth: float = 0.1,
    palette = None,
    use_raw: bool | None = None,
    log: bool = False,
    num_categories: int = 7,
    expression_cutoff: float = 0.0,
    mean_only_expressed: bool = False,
    standard_scale: Literal["var", "group"] | None = None,
    title: str | None = None,
    #colorbar_title: str | None = DotPlot.DEFAULT_COLOR_LEGEND_TITLE,
    #size_title: str | None = DotPlot.DEFAULT_SIZE_LEGEND_TITLE,
    figsize: tuple[float, float] | None = None,
    gene_symbols: str | None = None,
    var_group_positions: Sequence[tuple[int, int]] | None = None,
    var_group_labels: Sequence[str] | None = None,
    var_group_rotation: float | None = None,
    layer: str | None = None,
    dot_color_df: pd.DataFrame | None = None,
    show: bool | None = None,
    save: str | bool | None = None,
    ax: _AxesSubplot | None = None,
    return_fig: bool | None = False,
    vmin: float | None = None,
    vmax: float | None = None,
    vcenter: float | None = None,
    norm: Normalize | None = None,
    **kwds,
) -> RidgePlot | dict | None:


    # cmap = kwds.get("color_map", cmap)
    # if "color_map" in kwds:
    #     del kwds["color_map"]

    rp = RidgePlot(
        adata,
        var_names,
        groupby,
        bandwidth=bandwidth,
        use_raw=use_raw,
        log=log,
        num_categories=num_categories,
        expression_cutoff=expression_cutoff,
        mean_only_expressed=mean_only_expressed,
        standard_scale=standard_scale,
        title=title,
        figsize=figsize,
        gene_symbols=gene_symbols,
        var_group_positions=var_group_positions,
        var_group_labels=var_group_labels,
        var_group_rotation=var_group_rotation,
        layer=layer,
        dot_color_df=dot_color_df,
        ax=ax,
        vmin=vmin,
        vmax=vmax,
        vcenter=vcenter,
        norm=norm,
        palette=palette,
        **kwds,
    )

    if return_fig:
        return rp
    else:
        rp.make_figure()
        savefig_or_show(RidgePlot.DEFAULT_SAVE_PREFIX, show=show, save=save)
        show = settings.autoshow if show is None else show
        if not show:
            return rp.get_axes()





