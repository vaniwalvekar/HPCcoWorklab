"""Tests for hpclint.report (HPC-21): sacct usage parsing + utilization summary.
Pure logic only - run_sacct_usage is exercised on Libra (HPC-10)."""

from hpclint.report import parse_usage_line, summarize


def test_parse_usage_line_full():
    # JobID|JobName|Partition|AllocCPUS|TotalCPU|Elapsed|MaxRSS|ReqMem
    line = "123|myjob|compute|16|01:00:00|02:00:00|8000|16G"
    r = parse_usage_line(line)
    assert r["alloc_cpus"] == 16
    assert r["totalcpu_seconds"] == 3600
    assert r["elapsed_seconds"] == 7200
    assert r["maxrss_mb"] == 8000.0
    assert r["req_mem_gb"] == 16.0


def test_parse_usage_line_na_fields():
    r = parse_usage_line("1|j|c|N/A|N/A|00:10:00|N/A|N/A")
    assert r and r["alloc_cpus"] is None
    assert r["totalcpu_seconds"] is None
    assert r["maxrss_mb"] is None
    assert r["req_mem_gb"] is None


def test_parse_usage_line_malformed_returns_none():
    assert parse_usage_line("too|few|fields") is None
    assert parse_usage_line("") is None


def test_summarize_utilization_and_worst_ordering():
    # job 1: 16 cpus x 2h elapsed = 32 core-h asked, used TotalCPU 2h = 2 core-h -> 6.25%
    # job 2: 4 cpus x 4h elapsed = 16 core-h asked, used TotalCPU 4h -> 25%
    recs = [
        parse_usage_line("1|a|compute|16|02:00:00|02:00:00|8000|16G"),
        parse_usage_line("2|b|compute|4|04:00:00|04:00:00|8000|16G"),
    ]
    s = summarize(recs)
    assert s["jobs"] == 2
    assert s["cpu_alloc_core_hours"] == 32 + 16
    assert s["cpu_used_core_hours"] == 2 + 4
    assert 0 < s["cpu_utilization"] < 1
    assert s["worst"][0][1] == "1"   # lowest utilization first
    assert s["worst"][-1][1] == "2"


def test_summarize_empty_is_safe():
    s = summarize([])
    assert s["jobs"] == 0
    assert s["cpu_utilization"] is None
    assert s["worst"] == []


def test_summarize_skips_unusable_rows():
    # alloc_cpus None (N/A) -> contributes nothing, no crash
    recs = [parse_usage_line("1|a|compute|N/A|01:00:00|02:00:00|1000|4G")]
    s = summarize(recs)
    assert s["jobs"] == 1
    assert s["cpu_alloc_core_hours"] == 0
    assert s["worst"] == []
