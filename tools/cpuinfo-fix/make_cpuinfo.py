# Build a /proc/cpuinfo listing every *present* CPU, numbered with real sysfs IDs. CPUs hidden by the
# container are filled with a copy of the nearest visible CPU of the same cluster.
import re, sys
def parse_list(s):
    out = []
    for part in s.strip().split(","):
        a, _, b = part.partition("-")
        out += range(int(a), int(b or a) + 1)
    return out
sysfs = "/sys/devices/system/cpu"
online = parse_list(open(f"{sysfs}/online").read())
present = parse_list(open(f"{sysfs}/present").read())
text = open("/proc/cpuinfo").read()
head, _, _ = text.partition("processor")
blocks = [b for b in re.split(r"\n\s*\n", text.strip()) if b.startswith("processor")]
extra = "\n\n".join(b for b in re.split(r"\n\s*\n", text.strip()) if not b.startswith("processor"))
assert len(blocks) == len(online), (len(blocks), online)
by_id = dict(zip(online, blocks))
def cluster(c):
    return open(f"{sysfs}/cpu{c}/topology/cluster_cpus_list").read().strip()
out = []
for c in present:
    src = c if c in by_id else min((o for o in online if cluster(o) == cluster(c)), key=lambda o: abs(o - c))
    out.append(re.sub(r"^processor\s*:\s*\d+", f"processor\t: {c}", by_id[src], count=1))
sys.stdout.write("\n\n".join(out) + "\n\n" + (extra + "\n" if extra else ""))
