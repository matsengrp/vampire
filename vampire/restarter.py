import sys
import time

import click
import delegator
import pexpect


@click.command()
@click.argument('command')
def cli(command):
    """
    This includes doing filters as well as deduplicating on vjcdr3s.
    """

    max_n_retries = 10
    timeout = 60

    retry = 0
    click.echo(f"Running `{command}`.")

    while True:

        try:
            main_command = pexpect.spawn(command, logfile=sys.stdout.buffer)
            main_command.expect("scons: done building targets", timeout=timeout)
            click.echo("SCons has completed.")
            break
        except pexpect.TIMEOUT:
            click.echo("pexpect timed out.")
            pass
        except pexpect.EOF:
            click.echo("Process stopped without SCons completing.")

        while True:
            c = delegator.run('squeue -u matsen -M koshu | tail -n +3 | wc -l')
            jobs_remaining = int(c.out)
            if jobs_remaining == 0:
                break
            click.echo(f"{jobs_remaining} jobs still running.")
            time.sleep(10)

        if main_command.exitstatus != 0:
            click.echo("Exiting with an error.")
            sys.exit(main_command.exitstatus)

        if retry < max_n_retries:
            retry = retry + 1
            click.echo(f"Things seem stuck. Restarting for retry {retry} of {max_n_retries}...")


if __name__ == '__main__':
    cli()
