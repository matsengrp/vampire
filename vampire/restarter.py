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

    while(True):

        try:
            main_command = delegator.run(command)
            main_command.expect("scons: done building targets")
            click.echo("SCons has completed.")
            break
        except TIMEOUT:
            pass

        # Once we've timed out, check to see if it seems like things are still active.

        c = delegator.run('ps -eo user,comm,pcpu | grep matsen | grep python')
        python_total_cpu = sum([float(s.split()[-1]) for s in c.out.strip().split('\n')])
        if python_total_cpu > 1:
            click.echo("Python still active.")
            time.sleep(sleep_time)
            continue

        c = delegator.run('squeue -u matsen -M beagle | tail -n +3 | wc -l')
        if int(c.out) > 0:
            click.echo("Jobs still running.")
            time.sleep(sleep_time)
            continue

        click.echo("Things seem stuck. Last output:")
        click.echo(main_command.out.rstrip().split('\n')[-1])

        if retry < max_n_retries:
            click.echo(f"Restarting...")
            main_command.kill()
            main_command = delegator.run(command, block=False)
            retry = retry+1



if __name__ == '__main__':
    cli()
