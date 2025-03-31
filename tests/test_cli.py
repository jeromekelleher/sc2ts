import io
import json
import collections

import numpy as np
import click.testing as ct
import pytest
import tskit
import tomli_w
import pandas as pd

import sc2ts
from sc2ts import __main__ as main
from sc2ts import cli


class TestImportAlignments:

    def test_init(self, tmp_path, fx_alignments_fasta):
        ds_path = tmp_path / "ds.zarr"
        runner = ct.CliRunner(mix_stderr=False)
        result = runner.invoke(
            cli.cli,
            f"import-alignments {ds_path} {fx_alignments_fasta} -i --no-progress",
            catch_exceptions=False,
        )
        assert result.exit_code == 0

    def test_duplicate_aligments(self, tmp_path, fx_alignments_fasta):
        ds_path = tmp_path / "ds.zarr"
        runner = ct.CliRunner(mix_stderr=False)
        result = runner.invoke(
            cli.cli,
            f"import-alignments {ds_path} {fx_alignments_fasta} -i --no-progress",
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        result = runner.invoke(
            cli.cli,
            f"import-alignments {ds_path} {fx_alignments_fasta} --no-progress",
            catch_exceptions=True,
        )
        assert result.exit_code == 1


class TestImportMetadata:
    def test_suite_data(self, tmp_path, fx_metadata_tsv, fx_alignments_fasta):
        ds_path = tmp_path / "ds.zarr"
        runner = ct.CliRunner(mix_stderr=False)
        result = runner.invoke(
            cli.cli,
            f"import-alignments {ds_path} {fx_alignments_fasta} -i --no-progress",
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        fields_path = tmp_path / "fields.json"
        with open(fields_path, "w") as f:
            f.write(json.dumps({"NO SUCH": "A", "Viridian_pangolin": "PANGO"}))

        result = runner.invoke(
            cli.cli,
            f"import-metadata {ds_path} {fx_metadata_tsv} --field-descriptions={fields_path}",
            catch_exceptions=False,
        )
        ds = sc2ts.Dataset(ds_path)
        assert ds.metadata.fields["Viridian_pangolin"].attrs["description"] == "PANGO"

    def test_viridian_metadata(
        self, tmp_path, fx_raw_viridian_metadata_tsv, fx_alignments_fasta
    ):
        ds_path = tmp_path / "ds.zarr"
        runner = ct.CliRunner(mix_stderr=False)
        result = runner.invoke(
            cli.cli,
            f"import-alignments {ds_path} {fx_alignments_fasta} -i --no-progress",
            catch_exceptions=False,
        )
        assert result.exit_code == 0

        result = runner.invoke(
            cli.cli,
            f"import-metadata {ds_path} {fx_raw_viridian_metadata_tsv} --viridian",
            catch_exceptions=False,
        )


class TestRunHmm:

    def test_single_defaults(self, tmp_path, fx_ts_map, fx_dataset):
        strain = "ERR4206593"
        ts = fx_ts_map["2020-02-04"]
        ts_path = tmp_path / "ts.ts"
        ts.dump(ts_path)
        runner = ct.CliRunner(mix_stderr=False)
        result = runner.invoke(
            cli.cli,
            f"run-hmm {fx_dataset.path} {ts_path} {strain}",
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        lines = result.stdout.splitlines()
        assert len(lines) == 1
        d = json.loads(lines[0])
        assert d["strain"] == strain
        assert d["num_mismatches"] == 3
        assert d["direction"] == "forward"
        assert len(d["match"]["path"]) == 1
        assert len(d["match"]["mutations"]) == 5

    def test_multi_defaults(self, tmp_path, fx_ts_map, fx_dataset):
        copies = 10
        strains = ["ERR4206593"] * 10
        ts = fx_ts_map["2020-02-13"]
        ts_path = tmp_path / "ts.ts"
        ts.dump(ts_path)
        runner = ct.CliRunner(mix_stderr=False)
        result = runner.invoke(
            cli.cli,
            f"run-hmm {fx_dataset.path} {ts_path} " + " ".join(strains),
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        lines = result.stdout.splitlines()
        assert len(lines) == copies
        d = json.loads(lines[0])
        assert d["strain"] == strains[0]
        assert d["num_mismatches"] == 3
        assert d["direction"] == "forward"
        assert len(d["match"]["path"]) == 1
        assert len(d["match"]["mutations"]) == 0
        for line in lines[1:]:
            d2 = json.loads(line)
            assert d == d2

    def test_single_options(self, tmp_path, fx_ts_map, fx_dataset):
        strain = "ERR4206593"
        ts = fx_ts_map["2020-02-04"]
        ts_path = tmp_path / "ts.ts"
        ts.dump(ts_path)
        runner = ct.CliRunner(mix_stderr=False)
        result = runner.invoke(
            cli.cli,
            f"run-hmm {fx_dataset.path} {ts_path} {strain}"
            " --direction=reverse --num-mismatches=5 --num-threads=4",
            " --no-deletions-as-missing",
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        lines = result.stdout.splitlines()
        assert len(lines) == 1
        d = json.loads(lines[0])
        assert d["strain"] == strain
        assert d["num_mismatches"] == 5
        assert d["direction"] == "reverse"
        assert len(d["match"]["path"]) == 1
        assert len(d["match"]["mutations"]) == 5


class TestInfer:

    def make_config(
        self,
        tmp_path,
        dataset,
        run_id="test",
        results_dir="results",
        log_dir="logs",
        matches_dir="matches",
        exclude_sites=list(),
        override=list(),
        extra_top_level=dict(),
        **kwargs,
    ):
        config = {
            "dataset": str(dataset.path),
            "date_field": "date",
            "run_id": run_id,
            "results_dir": str(tmp_path / results_dir),
            "log_dir": str(tmp_path / log_dir),
            "matches_dir": str(tmp_path / matches_dir),
            "exclude_sites": exclude_sites,
            "extend_parameters": {**kwargs},
            "override": override,
            **extra_top_level,
        }
        filename = tmp_path / "config.toml"
        with open(filename, "w") as f:
            toml = tomli_w.dumps(config)
            # print("Generated", toml)
            f.write(toml)
        return filename

    def test_initialise_defaults(self, tmp_path, fx_dataset):
        config_file = self.make_config(tmp_path, fx_dataset)
        runner = ct.CliRunner(mix_stderr=False)
        result = runner.invoke(
            cli.cli,
            f"infer {config_file} --stop 2020-01-01",
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        init_ts_path = tmp_path / "results" / "test" / "test_init.ts"
        init_ts = tskit.load(init_ts_path)
        other_ts = sc2ts.initial_ts()
        other_ts.tables.assert_equals(init_ts.tables)
        match_db_path = tmp_path / "matches" / "test.matches.db"
        match_db = sc2ts.MatchDb(match_db_path)
        assert len(match_db) == 0

    @pytest.mark.parametrize("problematic", [[100], [100, 200]])
    def test_problematic_sites(self, tmp_path, fx_dataset, problematic):
        config_file = self.make_config(tmp_path, fx_dataset, exclude_sites=problematic)
        runner = ct.CliRunner(mix_stderr=False)
        result = runner.invoke(
            cli.cli,
            f"infer {config_file} --stop 2020-01-01",
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        init_ts_path = tmp_path / "results" / "test" / "test_init.ts"
        init_ts = tskit.load(init_ts_path)
        other_ts = sc2ts.initial_ts(problematic_sites=problematic)
        other_ts.tables.assert_equals(init_ts.tables)
        match_db_path = tmp_path / "matches" / "test.matches.db"
        match_db = sc2ts.MatchDb(match_db_path)
        assert len(match_db) == 0

    def test_first_day(self, tmp_path, fx_ts_map, fx_dataset):
        config_file = self.make_config(
            tmp_path, fx_dataset, exclude_sites=[56, 57, 58, 59, 60]
        )
        runner = ct.CliRunner(mix_stderr=False)
        result = runner.invoke(
            cli.cli,
            f"infer {config_file} --stop 2020-01-20",
            catch_exceptions=False,
        )
        date = "2020-01-19"
        assert result.exit_code == 0
        ts_path = tmp_path / "results" / "test" / f"test_{date}.ts"
        out_ts = tskit.load(ts_path)
        out_ts.tables.assert_equals(fx_ts_map[date].tables, ignore_provenance=True)

    def test_unknown_keys(self, tmp_path, fx_ts_map, fx_dataset):
        config_file = self.make_config(
            tmp_path, fx_dataset, extra_top_level={"Akey": 1, "B": 2}
        )
        runner = ct.CliRunner(mix_stderr=False)
        with pytest.raises(ValueError, match="Akey"):
            result = runner.invoke(
                cli.cli, f"infer {config_file} ", catch_exceptions=False
            )

    def test_unknown_params(self, tmp_path, fx_ts_map, fx_dataset):
        config_file = self.make_config(
            tmp_path,
            fx_dataset,
            no_such_param=1,
        )
        runner = ct.CliRunner(mix_stderr=False)
        with pytest.raises(TypeError, match="no_such_param"):
            result = runner.invoke(
                cli.cli, f"infer {config_file} ", catch_exceptions=False
            )

    def test_start(self, tmp_path, fx_ts_map, fx_dataset):
        config_file = self.make_config(
            tmp_path, fx_dataset, exclude_sites=[56, 57, 58, 59, 60]
        )
        runner = ct.CliRunner(mix_stderr=False)
        result = runner.invoke(
            cli.cli,
            f"infer {config_file} --stop 2020-01-20",
            catch_exceptions=False,
        )
        date = "2020-01-19"
        result = runner.invoke(
            cli.cli,
            f"infer {config_file} --start={date} --stop 2020-01-20 -f",
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        ts_path = tmp_path / "results" / "test" / f"test_{date}.ts"
        out_ts = tskit.load(ts_path)
        out_ts.tables.assert_equals(fx_ts_map[date].tables, ignore_provenance=True)

    def test_include_samples(self, tmp_path, fx_ts_map, fx_dataset):
        config_file = self.make_config(
            tmp_path,
            fx_dataset,
            exclude_sites=[56, 57, 58, 59, 60],
            include_samples=["SRR14631544", "NO_SUCH_STRAIN"],
        )
        runner = ct.CliRunner(mix_stderr=False)
        result = runner.invoke(
            cli.cli,
            f"infer {config_file} --stop 2020-01-02",
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        date = "2020-01-01"
        ts_path = tmp_path / "results" / "test" / f"test_{date}.ts"

        assert result.exit_code == 0
        ts = tskit.load(ts_path)
        assert "SRR14631544" in ts.metadata["sc2ts"]["samples_strain"]
        assert np.sum(ts.nodes_time[ts.samples()] == 0) == 1
        assert ts.num_samples == 1

    def test_override(self, tmp_path, fx_ts_map, fx_dataset):
        hmm_cost_threshold = 47
        config_file = self.make_config(
            tmp_path,
            fx_dataset,
            exclude_sites=[56, 57, 58, 59, 60],
            override=[
                {
                    "start": "2020-01-01",
                    "stop": "2020-01-02",
                    "parameters": {"hmm_cost_threshold": hmm_cost_threshold},
                }
            ],
        )
        runner = ct.CliRunner(mix_stderr=False)
        result = runner.invoke(
            cli.cli,
            f"infer {config_file} --stop 2020-01-02",
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        date = "2020-01-01"
        ts_path = tmp_path / "results" / "test" / f"test_{date}.ts"
        ts = tskit.load(ts_path)
        params = json.loads(ts.provenance(-1).record)["parameters"]
        assert params["hmm_cost_threshold"] == hmm_cost_threshold

        assert "SRR14631544" in ts.metadata["sc2ts"]["samples_strain"]
        assert np.sum(ts.nodes_time[ts.samples()] == 0) == 1
        assert ts.num_samples == 1

    def test_multiple_override(self, tmp_path, fx_ts_map, fx_dataset):
        hmm_cost_threshold = 3
        config_file = self.make_config(
            tmp_path,
            fx_dataset,
            exclude_sites=[56, 57, 58, 59, 60],
            # Overrides get applied sequentially, and last overlapping value wins.
            override=[
                {
                    "start": "2020-01-01",
                    "stop": "2020-01-02",
                    "parameters": {"hmm_cost_threshold": 123},
                },
                {
                    "start": "2020",
                    "stop": "2020-07-01",
                    "parameters": {"hmm_cost_threshold": hmm_cost_threshold},
                },
            ],
            hmm_cost_threshold=4000,
        )
        runner = ct.CliRunner(mix_stderr=False)
        result = runner.invoke(
            cli.cli,
            f"infer {config_file} --stop 2020-01-02",
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        date = "2020-01-01"
        ts_path = tmp_path / "results" / "test" / f"test_{date}.ts"
        ts = tskit.load(ts_path)
        params = json.loads(ts.provenance(-1).record)["parameters"]
        assert params["hmm_cost_threshold"] == hmm_cost_threshold

        assert "SRR14631544" not in ts.metadata["sc2ts"]["samples_strain"]
        assert ts.num_samples == 0


class TestPostprocess:

    def test_example(self, tmp_path, fx_ts_map, fx_match_db):
        ts = fx_ts_map["2020-02-13"]
        out_ts_path = tmp_path / "ts.ts"
        runner = ct.CliRunner(mix_stderr=False)
        result = runner.invoke(
            cli.cli,
            f"postprocess {ts.path} {out_ts_path} --match-db={fx_match_db.path}",
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        out = tskit.load(out_ts_path)
        assert out.num_samples == ts.num_samples + 8
        assert out.num_provenances == ts.num_provenances + 3


class TestMapDeletions:

    def test_example(self, tmp_path, fx_ts_map, fx_dataset):
        ts = fx_ts_map["2020-02-13"]
        out_ts_path = tmp_path / "ts.ts"
        runner = ct.CliRunner(mix_stderr=False)
        result = runner.invoke(
            cli.cli,
            f"map-deletions {fx_dataset.path} {ts.path} {out_ts_path} "
            "--frequency-threshold=0.0001",
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        out = tskit.load(out_ts_path)
        remapped_sites = [
            j
            for j in range(ts.num_sites)
            if "original_mutations" in out.site(j).metadata["sc2ts"]
        ]
        assert remapped_sites == [1541, 3945, 3946, 3947]


class TestValidate:

    @pytest.mark.parametrize("date", ["2020-01-01", "2020-02-11"])
    def test_date(self, tmp_path, fx_ts_map, fx_dataset, date):
        ts = fx_ts_map[date]
        ts_path = tmp_path / "ts.ts"
        ts.dump(ts_path)
        runner = ct.CliRunner(mix_stderr=False)
        result = runner.invoke(
            cli.cli,
            f"validate {fx_dataset.path} {ts_path} --date-field=date",
            catch_exceptions=False,
        )
        assert result.exit_code == 0


class TestInfoTs:
    def test_example(self, fx_ts_map):
        ts = fx_ts_map["2020-02-13"]
        runner = ct.CliRunner(mix_stderr=False)
        result = runner.invoke(
            cli.cli,
            f"info-ts {ts.path}",
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        assert "latest_sample" in result.stdout


class TestInfoMatches:
    def test_defaults(self, fx_match_db):
        runner = ct.CliRunner(mix_stderr=False)
        result = runner.invoke(
            cli.cli,
            f"info-matches {fx_match_db.path}",
            catch_exceptions=False,
        )
        assert result.exit_code == 0

    def test_all_matches(self, fx_match_db):
        runner = ct.CliRunner(mix_stderr=False)
        result = runner.invoke(
            cli.cli,
            f"info-matches -A {fx_match_db.path}",
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        df = pd.read_csv(io.StringIO(result.stdout), sep="\t")
        assert list(df) == [
            "strain",
            "n_parents",
            "n_mutations",
            "parents",
            "mutations",
        ]
        assert df.shape[0] == 55
        assert np.all(df["n_parents"] == 1)
        assert df["n_mutations"].values[0] == 26


class TestInfoDataset:
    def test_defaults(self, fx_dataset):
        runner = ct.CliRunner(mix_stderr=False)
        result = runner.invoke(
            cli.cli,
            f"info-dataset {fx_dataset.path}",
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        assert "with 55 samples and 26 metadata fields" in result.stdout

    def test_zarr(self, fx_dataset):
        runner = ct.CliRunner(mix_stderr=False)
        result = runner.invoke(
            cli.cli,
            f"info-dataset {fx_dataset.path} -z",
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        # Pick arbitrary field as a basic check
        assert "/sample_Genbank_N" in result.stdout
