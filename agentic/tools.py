def apply_strategy(ds, strategy):
    return ds.to_netcdf("compressed.nc", encoding=strategy)
