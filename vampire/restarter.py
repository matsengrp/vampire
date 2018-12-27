import re
import time

import click
import delegator


@click.command()
@click.argument('command')
def cli(command):
    """
    This includes doing filters as well as deduplicating on vjcdr3s.
    """

    retry=0
    max_n_retries = 100
    sleep_time=10

    click.echo(f"Running `{command}`.")

    while True:

        try:
            main_command = delegator.run(command)
            main_command.expect("scons: done building targets")
            click.echo("SCons has completed.")
            break
        except TIMEOUT:
            pass

        # Once we've timed out, check to see if it seems like things are still active.

        jobs_remaining = 1

        while jobs_remaining > 0:
            c = delegator.run('squeue -u matsen -M beagle | tail -n +3 | wc -l')
            jobs_remaining = int(c.out)
            click.echo("{jobs_remaining} jobs still running.")
            time.sleep(sleep_time)
            continue

        click.echo("Things seem stuck. Last output:")
        click.echo(main_command.out.rstrip().split('\n')[-1])

        if retry < max_n_retries:
            retry = retry+1
            click.echo(f"Restarting for retry {retry} of {max_n_retries}...")


if __name__ == '__main__':
    cli()
