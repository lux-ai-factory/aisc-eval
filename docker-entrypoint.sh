#!/bin/sh
# The eval worker's start. As root, before the worker: the folders the run users' locks and the egress
# proxy's rules live in, the firewall that keeps every run user (uids 10100-10199) behind the proxy
# (aisc_eval/egress.py), and the proxy itself, restarted if it ends. Without NET_ADMIN the firewall cannot be
# set: the worker then refuses to start plugin tests and says why, and other commands (Flower) run as usual.
set -u

if [ "$(id -u)" = "0" ]; then
    mkdir -p /run/aisc-run-users /run/aisc-egress
    chmod 0700 /run/aisc-run-users
    rm -f /run/aisc-egress/*.json /run/aisc-egress/*.tmp
    chown root:aisc-egress /run/aisc-egress && chmod 0750 /run/aisc-egress
    if python -m aisc_eval.egress; then
        (
            while true; do
                python -m aisc_eval.egress_proxy
                echo "egress proxy ended; starting it again" >&2
                sleep 1
            done
        ) &
    else
        echo "the plugin firewall could not be set (iptables, NET_ADMIN): plugin tests will be refused" >&2
    fi
fi

exec "$@"
