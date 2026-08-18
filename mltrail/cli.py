"""Command-line interface for MLTrail: a thin argparse wrapper over the Registry API.

Modes (mutually exclusive): --add, --overwrite, --predict, --list, --details, --search, --trail.
Every mode is a one-to-one call into the same core used from notebooks.
"""
import argparse
import sys

import pandas as pd

from .config import default_config, load_config
from .registry import Registry
from .schema import IDENTITY_FIELDS, ValidationError

# Registry metadata fields settable via CLI flags (model_path/comment handled explicitly below).
_ENTRY_FIELDS = IDENTITY_FIELDS + ["dataset_path", "df_pred_path"]


def _format_table(df, max_col_width=60):
    """Render a DataFrame as a bordered box table for the terminal.

    Numeric columns are right-aligned, everything else left-aligned; NaN prints blank. Cells
    wider than `max_col_width` are clipped with an ellipsis for readability (display only — the
    DataFrame keeps the full value; use --details for the full text). Empty frame -> placeholder.
    """
    if df.empty:
        return "(no matching models)"

    def clip(s):
        return s if len(s) <= max_col_width else s[:max_col_width - 1] + "…"

    cols = [clip(str(c)) for c in df.columns]
    rows = [[clip("" if pd.isna(v) else str(v)) for v in row] for row in df.itertuples(index=False)]
    widths = [max(len(cols[i]), *(len(r[i]) for r in rows)) for i in range(len(cols))]
    align = [str.rjust if pd.api.types.is_numeric_dtype(df[c]) else str.ljust for c in df.columns]

    def line(left, mid, right):
        return left + mid.join("─" * (w + 2) for w in widths) + right

    def row(cells):
        return "│ " + " │ ".join(align[i](cells[i], widths[i]) for i in range(len(cols))) + " │"

    return "\n".join([line("┌", "┬", "┐"), row(cols), line("├", "┼", "┤"),
                      *(row(r) for r in rows), line("└", "┴", "┘")])


def build_parser():
    p = argparse.ArgumentParser(prog="mltrail", description="Local versioned ML model registry (vault).")
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--add", action="store_true", help="register a model (no --id) or add a version (--id)")
    mode.add_argument("--overwrite", action="store_true", help="reset the latest version of --id in place")
    mode.add_argument("--predict", action="store_true", help="predict on a dataset with model --id")
    mode.add_argument("--list", action="store_true", help="list models (id, date, experiment_name, measure)")
    mode.add_argument("--details", action="store_true", help="show every attribute of model --id")
    mode.add_argument("--search", action="store_true", help="list models matching ANY given field")
    mode.add_argument("--trail", action="store_true", help="track a metric across versions")
    mode.add_argument("--delete", action="store_true", help="delete model --id and its entire version trail")
    mode.add_argument("--save-trainset", dest="save_trainset", action="store_true",
                      help="archive model --id's training set (--dataset), storing only new rows")

    p.add_argument("--config", help="path to config.yaml (default: MLTrail's own config/config.yaml, wherever the package lives)")
    p.add_argument("--id", type=int, help="model id")

    for f in _ENTRY_FIELDS:
        p.add_argument(f"--{f}")
    p.add_argument("--model_path",
                   help="add: source model artifact to import into the vault (file or dir); "
                        "predict: path override for the artifact to load")
    p.add_argument("--comment",
                   help="add: free-text note documenting what this model/version does "
                        "(stored per version, shown by --details)")
    p.add_argument("--metrics", nargs="+", metavar="NAME=VALUE",
                   help="add: e.g. --metrics R2=0.81 mse=0.12 ; trail: a single metric name")

    p.add_argument("--training_set",
                   help="add: training-set path to archive (sliced to compound_id/smiles/label)")
    p.add_argument("--label_column", help="add: name of the label/target column in --training_set")
    p.add_argument("--dataset", help="predict / save-trainset: dataset path (csv/tsv/excel/sdf/parquet)")
    p.add_argument("--dedup_on", nargs="+", metavar="COL",
                   help="save-trainset: columns identifying a row (default: all columns)")
    p.add_argument("--smiles_column", help="predict: name of the SMILES column")
    p.add_argument("--compound_id", help="predict: name of the compound-id column (or n/a)")
    p.add_argument("--pred_output", help="predict: CSV path to write predictions to")
    p.add_argument("--output_trail", help="trail: CSV path to write the metric-vs-date table to")
    return p


def _resolve_config(path):
    # --config wins; otherwise MLTrail's own package-relative config (default_config), so the
    # same vault resolves from any working directory (not the CWD's config.yaml).
    return load_config(path) if path else default_config()


def _parse_metrics(items):
    """Parse ['R2=0.81', 'mse=0.12'] into {'R2': 0.81, 'mse': 0.12}; numeric values become floats."""
    metrics = {}
    for item in items:
        name, _, value = item.partition("=")
        try:
            metrics[name] = float(value)
        except ValueError:
            metrics[name] = value
    return metrics


def _cmd_add(reg, args):
    if args.overwrite and args.id is None:
        raise ValidationError("--overwrite requires --id")
    fields = {f: getattr(args, f) for f in _ENTRY_FIELDS}
    fields["comment"] = args.comment
    if args.metrics:
        fields["metrics"] = _parse_metrics(args.metrics)
    model_id = reg.add(
        model_id=args.id, overwrite=args.overwrite,
        model=args.model_path, training_set=args.training_set, label_column=args.label_column,
        smiles_column=args.smiles_column or "smiles",
        compound_id_column=args.compound_id or "compound_id",
        **fields,
    )
    action = "overwritten" if args.overwrite else ("new version added" if args.id else "added")
    print(f"model {model_id}: {action}")


def _cmd_predict(reg, args):
    out = reg.predict(args.id, args.dataset, smiles_column=args.smiles_column,
                      compound_id=args.compound_id, pred_output=args.pred_output,
                      model_path=args.model_path)
    n_failed = int(out[[c for c in out.columns if c not in ("smiles", "compound")]].isna().all(axis=1).sum())
    if args.pred_output:
        note = f" ({n_failed} rows had unparseable SMILES)" if n_failed else ""
        print(f"wrote {len(out)} predictions -> {args.pred_output}{note}")
    else:
        print(out.to_string(index=False))


def _cmd_details(reg, args):
    for key, value in reg.details(args.id).items():
        print(f"- {key} = {value}")


def _cmd_search(reg, args):
    filters = {f: getattr(args, f) for f in IDENTITY_FIELDS if getattr(args, f) is not None}
    print(_format_table(reg.search(**filters)))


def _cmd_trail(reg, args):
    metric = args.metrics[0] if args.metrics else None
    if metric is None:
        raise ValidationError("--trail requires --metrics <metric name>")
    df = reg.trail(metric, model_id=args.id)
    if args.output_trail:
        df.to_csv(args.output_trail, index=False)
        print(f"wrote trail ({len(df)} rows) -> {args.output_trail}")
    else:
        print(_format_table(df))


def _cmd_delete(reg, args):
    if args.id is None:
        raise ValidationError("--delete requires --id")
    reg.delete(args.id)
    print(f"model {args.id}: deleted (all versions)")


def _cmd_save_trainset(reg, args):
    if args.id is None:
        raise ValidationError("--save-trainset requires --id")
    summary = reg.save_training_set(args.id, args.dataset, dedup_on=args.dedup_on)
    if summary["chunk"]:
        print(f"model {args.id}: +{summary['n_new']} new rows "
              f"(total {summary['n_total']}) -> {summary['chunk']}")
    else:
        print(f"model {args.id}: no new rows ({summary['n_existing']} already stored)")


def main(argv=None):
    args = build_parser().parse_args(argv)
    reg = Registry.from_config(_resolve_config(args.config))
    try:
        if args.add or args.overwrite:
            _cmd_add(reg, args)
        elif args.predict:
            _cmd_predict(reg, args)
        elif args.list:
            print(_format_table(reg.list()))
        elif args.details:
            _cmd_details(reg, args)
        elif args.search:
            _cmd_search(reg, args)
        elif args.trail:
            _cmd_trail(reg, args)
        elif args.delete:
            _cmd_delete(reg, args)
        elif args.save_trainset:
            _cmd_save_trainset(reg, args)
    except (ValidationError, KeyError, ValueError, FileNotFoundError, NotImplementedError) as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
