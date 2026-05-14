import click

from smokesight.background import background
from smokesight.calibrate import calibrate
from smokesight.dynamics import dynamics
from smokesight.io import to_netcdf
from smokesight.retrieve import retrieve


@click.group()
def main():
    pass


@main.command()
@click.argument("video")
@click.argument("config")
@click.option("-o", "output", required=True)
@click.option("--n-frames", default=100)
@click.option("--no-dynamics", is_flag=True)
def run(video, config, output, n_frames, no_dynamics):
    try:
        cal = calibrate(video, config)
        bg = background(cal, n_frames=min(n_frames, cal.L.shape[0]))
        ret = retrieve(cal, bg)
        if not no_dynamics:
            dynamics(ret)
        to_netcdf(ret, output)
    except Exception as exc:
        click.echo(str(exc), err=True)
        raise SystemExit(1)


@main.command()
@click.argument("video")
@click.argument("config")
@click.option("-o", "output", required=True)
def calibrate_cmd(video, config, output):
    to_netcdf(calibrate(video, config), output)


@main.command()
@click.argument("output")
def inspect(output):
    import xarray as xr
    ds = xr.open_dataset(output)
    click.echo(str(ds))
