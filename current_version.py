from flask import Flask, request, render_template, send_file, jsonify
import pandas as pd, numpy as np, os, uuid, io, importlib, openpyxl, json
from werkzeug.utils import secure_filename
import threading, webbrowser
import sys, pathlib

BASE_DIR = pathlib.Path(getattr(sys, "_MEIPASS",
                                pathlib.Path(__file__).parent))

TEMPLATE_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

app = Flask(__name__,
            template_folder=str(TEMPLATE_DIR),
            static_folder=str(STATIC_DIR),
            static_url_path="/static")

TPL = str(TEMPLATE_DIR)
UPLOAD = str(UPLOAD_DIR)

CAT = {"categorical", "category", "binary", "boolean", "bool"}
NUM = {"continuous", "numeric", "number", "scale", "metric"}

if importlib.util.find_spec("xlsxwriter"):
    XL = "xlsxwriter"
elif importlib.util.find_spec("openpyxl"):
    XL = "openpyxl"
else:
    XL = None

    # â”€â”€â”€â”€â”€â”€â”€â”€â”€ helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€


def canon(v):
    try:
        f = float(v)
        return str(int(f)) if f.is_integer() else str(f)
    except:
        return str(v)


def _save(up):
    fn = f"{uuid.uuid4().hex}_{secure_filename(up.filename)}"
    up.save(os.path.join(UPLOAD, fn))
    return fn


def _allowed(fn):
    return "." in fn and fn.rsplit('.', 1)[1].lower() in {"csv"}


def _read(path):
    ext = path.rsplit('.', 1)[1].lower()
    if ext == 'csv':
        for enc in ('utf-8-sig', 'utf-8', 'cp1252', 'latin1'):
            try:
                return pd.read_csv(path, encoding=enc)
            except UnicodeDecodeError:
                pass
        raise
    return pd.read_excel(path, engine='openpyxl' if ext == 'xlsx' else None)


def _safe_sheet(name):
    safe = name.replace("/", "-").replace("\\", "-").replace("*", "").replace("[", "").replace("]", "").replace(":",
                                                                                                                "-")
    return safe[:31]


def _maps(mdf):
    vmap, lorder = {}, {}
    for col, g in mdf.groupby("column_name", sort=False):
        if {"value", "value_label"}.issubset(g.columns):
            codes = g["value"].apply(canon).tolist()
            labs = g["value_label"].tolist()
            vmap[col] = dict(zip(codes, labs))
            lorder[col] = labs

    qtext = (mdf.drop_duplicates("column_name")
             .set_index("column_name")["question_text"].to_dict())

    rankgrp, msgrp = {}, {}
    if "group" in mdf.columns:
        for grp, sub in mdf[mdf["question_type"].str.lower() == "ranking"] \
                .groupby("group", sort=False):
            rankgrp[grp] = sub["column_name"].tolist()

        for grp, sub in mdf[mdf["question_type"].str.lower() == "multi_select"] \
                .groupby("group", sort=False):
            msgrp[grp] = sub["column_name"].tolist()

        mask = (mdf['question_type'].str.lower().isin(['boolean', 'binary'])) \
               & mdf['group'].notna()
        for grp, sub in mdf[mask].groupby('group', sort=False):
            msgrp.setdefault(grp, []).extend(sub['column_name'].tolist())

    return vmap, lorder, qtext, rankgrp, msgrp


def calculate_question_statistics(df, mdf):
    """Calculate statistics for all questions in the dataframe"""
    mdf['qt'] = mdf['question_type'].str.lower().str.strip()
    qtypes = mdf.set_index("column_name")["qt"].to_dict()
    vmap, lorder, _, _, _ = _maps(mdf)

    stats = {}

    for col in df.columns:
        if col not in qtypes:
            continue

        qtype = qtypes[col]

        if qtype in NUM:
            # Numeric statistics
            numeric_col = pd.to_numeric(df[col], errors='coerce')
            stats[col] = {
                'min': float(numeric_col.min()) if not numeric_col.isna().all() else 0,
                'max': float(numeric_col.max()) if not numeric_col.isna().all() else 100,
                'mean': float(numeric_col.mean()) if not numeric_col.isna().all() else 50,
                'median': float(numeric_col.median()) if not numeric_col.isna().all() else 50,
                'type': 'numeric'
            }
        else:
            # Categorical distribution - canonicalize values first
            df_col_canon = df[col].apply(canon)
            value_counts = df_col_canon.value_counts(normalize=True, dropna=False)
            distribution = {}

            if col in lorder:
                # For each label in order, find its percentage
                for label in lorder[col]:
                    # Find the code for this label
                    code = None
                    if col in vmap:
                        for k, v in vmap[col].items():
                            if v == label:
                                code = k
                                break

                    if code and code in value_counts.index:
                        distribution[label] = float(value_counts[code] * 100)
                    else:
                        # Check if the label itself is in the counts
                        label_canon = canon(label)
                        if label_canon in value_counts.index:
                            distribution[label] = float(value_counts[label_canon] * 100)
                        else:
                            distribution[label] = 0.0
            else:
                # No predefined labels, use actual values
                for val, pct in value_counts.items():
                    distribution[str(val)] = float(pct * 100)

            stats[col] = {
                'distribution': distribution,
                'type': 'categorical'
            }

    return stats


# ───────── BENCHMARK FUNCTIONS ─────────
def calculate_benchmark_score(row, config, label_maps=None):
    """Calculate benchmark score for a single row"""
    total_score = 0
    total_weight = 0
    max_possible_score = 0
    max_possible_weight = 0

    # helper: canonicalize coded values like "3.0" -> "3"
    def _canon_code(v):
        s = str(v).strip()
        try:
            f = float(s)
            if f.is_integer():
                return str(int(f))
        except Exception:
            pass
        return s

    for question in config['questions']:
        value = row.get(question['column'])
        weight = question.get('weight', 1.0)

        # Calculate max possible for this question
        max_points = 0
        if question['type'] == 'numeric':
            if question['scoring']['method'] == 'direct':
                # For direct scoring, we need to estimate a reasonable max
                max_points = 100
            elif question['scoring']['method'] == 'binned':
                # Find the maximum points from bins
                max_points = max(bin_def['points'] for bin_def in question['scoring']['bins'])
        elif question['type'] == 'categorical':
            # Find the maximum points from the map
            mp = question['scoring'].get('map', {})
            max_points = max(mp.values()) if mp else 0

        max_possible_score += max_points * weight
        max_possible_weight += weight

        # Calculate actual score for this row
        if pd.isna(value):
            continue

        points = 0

        if question['type'] == 'numeric':
            try:
                num_val = float(value)
                if question['scoring']['method'] == 'direct':
                    points = num_val
                elif question['scoring']['method'] == 'binned':
                    # Find the appropriate bin
                    for bin_def in question['scoring']['bins']:
                        if num_val <= bin_def['max']:
                            points = bin_def['points']
                            break
            except (ValueError, TypeError):
                continue

        elif question['type'] == 'categorical':
            value_str = str(value).strip()
            cmap = question['scoring'].get('map', {})

            # 1) direct lookup (exact key)
            points = cmap.get(value_str, 0)

            # 2) case-insensitive match on label
            if points == 0 and cmap:
                vlower = value_str.lower()
                for key, val in cmap.items():
                    if str(key).strip().lower() == vlower:
                        points = val
                        break

            # 3) code -> label translation using label_maps[column], then lookup by label
            if points == 0 and label_maps and question['column'] in label_maps:
                code_key = _canon_code(value)
                label = label_maps.get(question['column'], {}).get(code_key)
                if label is not None:
                    label_str = str(label).strip()
                    # try exact label
                    points = cmap.get(label_str, 0)
                    # case-insensitive label match
                    if points == 0 and cmap:
                        llower = label_str.lower()
                        for key, val in cmap.items():
                            if str(key).strip().lower() == llower:
                                points = val
                                break

        total_score += points * weight
        total_weight += weight

    # Normalize score to 0-100 scale
    if total_weight == 0 or max_possible_weight == 0:
        return 0, "Unclassified"

    # Calculate percentage of maximum possible score
    if max_possible_score > 0:
        normalized_score = (total_score / max_possible_score) * 100
    else:
        # Fallback to weighted average if we can't determine max
        normalized_score = total_score / total_weight

    # Assign tier - check which tier the score falls into
    assigned_tier = "Unclassified"
    for tier in config['tiers']:
        if tier['min'] <= normalized_score <= tier['max']:
            assigned_tier = tier['label']
            break

    return round(normalized_score, 2), assigned_tier



def apply_benchmark_to_dataframe(df, config, label_maps=None):
    """Apply benchmark to entire dataframe"""
    scores = []
    tiers = []

    for idx, row in df.iterrows():
        score, tier = calculate_benchmark_score(row, config, label_maps)
        scores.append(score)
        tiers.append(tier)

    return pd.Series(scores, index=df.index), pd.Series(tiers, index=df.index)


@app.route('/', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        d = request.files.get('data_file')
        m = request.files.get('map_file')
        if not d or not m:
            return render_template('upload.html', error='Both files required.')
        if not (_allowed(d.filename) and _allowed(m.filename)):
            return render_template('upload.html', error='CSV Only.')
        try:
            dfn, mfn = _save(d), _save(m)
            df, mdf = _read(os.path.join(UPLOAD, dfn)), _read(os.path.join(UPLOAD, mfn))
        except Exception as e:
            return render_template('upload.html', error=str(e))
        if {'column_name', 'question_text', 'question_type'}.difference(mdf.columns):
            return render_template('upload.html', error='Data-map missing required columns.')

        # Store in session-like way (you might want to use Flask sessions)
        return render_template('benchmark_choice.html', data_fn=dfn, map_fn=mfn)

    return render_template('upload.html')


@app.route('/benchmark/setup', methods=['GET', 'POST'])
def benchmark_setup():
    """Step 1: Choose to build benchmark or skip"""
    if request.method == 'GET':
        dfn = request.args.get('data_fn')
        mfn = request.args.get('map_fn')
        build_benchmark = request.args.get('build_benchmark', 'no')
    else:
        dfn = request.form.get('data_fn')
        mfn = request.form.get('map_fn')
        build_benchmark = request.form.get('build_benchmark')

    if build_benchmark == 'no':
        # Skip to variable selection
        df = _read(os.path.join(UPLOAD, dfn))
        mdf = _read(os.path.join(UPLOAD, mfn))
        mdf['qt'] = mdf['question_type'].str.lower().str.strip()

        cat_df = mdf.loc[mdf['qt'].isin(CAT), ['column_name', 'question_text']].drop_duplicates()
        all_df = mdf[['column_name', 'question_text']].drop_duplicates()
        cat_df['question_text'] = cat_df['question_text'].fillna('').astype(str)
        all_df['question_text'] = all_df['question_text'].fillna('').astype(str)

        cat_opts = [(r.column_name, f"{r.column_name} - {r.question_text[:60]}")
                    for r in cat_df.itertuples(False)]
        all_opts = [(r.column_name, f"{r.column_name} - {r.question_text[:60]}")
                    for r in all_df.itertuples(False)]

        return render_template('select.html', data_fn=dfn, map_fn=mfn,
                               cat_opts=cat_opts, all_opts=all_opts)

    # Build benchmark - go to config page
    num_tiers = int(request.form.get('num_tiers', 3))
    benchmark_name = request.form.get('benchmark_name', 'New Benchmark')

    # Parse tier definitions
    tiers = []
    for i in range(num_tiers):
        tier_label = request.form.get(f'tier_{i}_label', f'Tier {i + 1}')
        tier_min = float(request.form.get(f'tier_{i}_min', i * (100 / num_tiers)))
        tier_max = float(request.form.get(f'tier_{i}_max', (i + 1) * (100 / num_tiers)))
        tiers.append({
            'label': tier_label,
            'min': tier_min,
            'max': tier_max
        })

    return render_template('benchmark_config.html',
                           data_fn=dfn,
                           map_fn=mfn,
                           num_tiers=num_tiers,
                           benchmark_name=benchmark_name,
                           tiers=tiers,
                           enumerate=enumerate)


# REPLACE the existing /benchmark/builder route with this updated version
@app.route('/benchmark/builder', methods=['GET', 'POST'])
def benchmark_builder():
    """Step 2: Build the benchmark"""
    if request.method == 'GET':
        dfn = request.args.get('data_fn')
        mfn = request.args.get('map_fn')
        num_tiers = int(request.args.get('num_tiers', 3))
        benchmark_name = request.args.get('benchmark_name', 'New Benchmark')

        # Get tier configuration from query params
        tiers = []
        for i in range(num_tiers):
            tier_label = request.args.get(f'tier_{i}_label', f'Tier {i + 1}')
            tier_min = float(request.args.get(f'tier_{i}_min', i * (100 / num_tiers)))
            tier_max = float(request.args.get(f'tier_{i}_max', (i + 1) * (100 / num_tiers)))
            tiers.append({
                'label': tier_label,
                'min': tier_min,
                'max': tier_max
            })
    else:
        dfn = request.form.get('data_fn')
        mfn = request.form.get('map_fn')
        num_tiers = int(request.form.get('num_tiers', 3))
        benchmark_name = request.form.get('benchmark_name', 'New Benchmark')

        # Parse tier definitions from form
        tiers = []
        for i in range(num_tiers):
            tier_label = request.form.get(f'tier_{i}_label', f'Tier {i + 1}')
            tier_min = float(request.form.get(f'tier_{i}_min', i * (100 / num_tiers)))
            tier_max = float(request.form.get(f'tier_{i}_max', (i + 1) * (100 / num_tiers)))
            tiers.append({
                'label': tier_label,
                'min': tier_min,
                'max': tier_max
            })

    # Load data to get available questions
    df = _read(os.path.join(UPLOAD, dfn))
    mdf = _read(os.path.join(UPLOAD, mfn))
    mdf['qt'] = mdf['question_type'].str.lower().str.strip()

    vmap, lorder, qtext, _, _ = _maps(mdf)
    qtypes = mdf.set_index("column_name")["qt"].to_dict()

    # Calculate statistics for all questions
    question_stats = calculate_question_statistics(df, mdf)

    # Prepare question options
    questions_data = []
    for col in df.columns:
        if col in qtypes:
            qtype = qtypes[col]
            q_info = {
                'column': col,
                'text': qtext.get(col, col),
                'type': 'numeric' if qtype in NUM else 'categorical',
                'options': lorder.get(col, []) if col in lorder else None
            }
            questions_data.append(q_info)

    return render_template('benchmark_builder.html',
                           data_fn=dfn,
                           map_fn=mfn,
                           benchmark_name=benchmark_name,
                           tiers=tiers,
                           questions=questions_data,
                           question_stats=question_stats,  # Now passing the stats
                           num_respondents=len(df))


# Updated preview_distribution endpoint with live score stats
@app.route('/api/preview_distribution', methods=['POST'])
def preview_distribution():
    """API endpoint to calculate distribution preview with score statistics"""
    try:
        data = request.json
        dfn = data['data_fn']
        config = data['benchmark_config']
        mfn = data.get('map_fn')  # <-- allow label map

        # Load data
        df = _read(os.path.join(UPLOAD, dfn))

        # Build code->label maps per column (optional)
        label_maps = {}
        try:
            if mfn:
                mdf = _read(os.path.join(UPLOAD, mfn))
                vmap, lorder, qtext, rankgrp, msgrp = _maps(mdf)  # vmap: {column: {code: label}}
                label_maps = vmap or {}
        except Exception:
            label_maps = {}

        # Calculate scores and tiers
        scores_list = []
        tiers_list = []
        for _, row in df.iterrows():
            score, tier = calculate_benchmark_score(row, config, label_maps=label_maps)
            scores_list.append(score)
            tiers_list.append(tier)

        tier_series = pd.Series(tiers_list, index=df.index) if len(df) else pd.Series(dtype=object)
        score_series = pd.Series(scores_list, index=df.index) if len(df) else pd.Series(dtype=float)

        # Count distribution
        distribution = tier_series.value_counts().to_dict() if len(tier_series) else {}
        total = int(len(df))

        # Order tiers by config
        result = []
        for tier in config['tiers']:
            count = int(distribution.get(tier['label'], 0))
            pct = (count / total * 100) if total > 0 else 0
            result.append({
                'label': tier['label'],
                'count': count,
                'percentage': round(pct, 1)
            })

        # Score stats
        if len(score_series):
            score_stats = {
                'min': float(score_series.min()),
                'max': float(score_series.max()),
                'mean': float(score_series.mean()),
                'median': float(score_series.median()),
                'std': float(score_series.std()) if len(score_series) > 1 else 0.0
            }
        else:
            score_stats = {'min': 0.0, 'max': 0.0, 'mean': 0.0, 'median': 0.0, 'std': 0.0}

        return jsonify({
            'success': True,
            'distribution': result,
            'total': total,
            'score_stats': score_stats
        })

    except Exception as e:
        import traceback
        print(f"ERROR in preview_distribution: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)})



# REPLACE the /benchmark/save route with this updated version that saves as ZIP with CSVs
@app.route('/benchmark/save', methods=['POST'])
def save_benchmark():
    """Save benchmark and create enhanced data file - saves as separate CSVs in ZIP"""
    try:
        dfn = request.form.get('data_fn')
        mfn = request.form.get('map_fn')
        config_json = request.form.get('benchmark_config')
        config = json.loads(config_json)

        # Load original data and map
        df = _read(os.path.join(UPLOAD, dfn))
        mdf = _read(os.path.join(UPLOAD, mfn)) if mfn else pd.DataFrame()

        # Build label maps (code -> label) for scoring
        label_maps = {}
        try:
            if not mdf.empty:
                vmap, lorder, qtext, rankgrp, msgrp = _maps(mdf)
                label_maps = vmap or {}
        except Exception:
            label_maps = {}

        # Apply benchmark
        score_col = f"{config['name']}_Score"
        tier_col  = f"{config['name']}_Tier"

        scores, tiers = apply_benchmark_to_dataframe(df, config, label_maps=label_maps)
        df[score_col] = scores
        df[tier_col]  = tiers

        # Extend data map with new fields
        new_rows = []

        # Score column (numeric)
        new_rows.append({
            'column_name': score_col,
            'question_text': f"Benchmark Score: {config['name']}",
            'question_type': 'numeric',
            'value': '',
            'value_label': ''
        })

        # Tier column (categorical with labels)
        for tier in config['tiers']:
            new_rows.append({
                'column_name': tier_col,
                'question_text': f"Benchmark Tier: {config['name']}",
                'question_type': 'categorical',
                'value': tier['label'],
                'value_label': tier['label']
            })

        mdf_new = pd.concat([mdf, pd.DataFrame(new_rows)], ignore_index=True)

        # Create a ZIP with data CSV, map CSV, and config JSON
        import zipfile
        from io import BytesIO

        zip_buffer = BytesIO()
        safe_name = config['name'].replace(' ', '_')

        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            # Data with benchmark
            zip_file.writestr(
                f"data_with_{safe_name}_benchmark.csv",
                df.to_csv(index=False)
            )
            # Updated data map
            zip_file.writestr(
                f"datamap_with_{safe_name}_benchmark.csv",
                mdf_new.to_csv(index=False)
            )
            # Benchmark config JSON
            zip_file.writestr(
                f"{safe_name}_benchmark_config.json",
                json.dumps(config, indent=2)
            )

        zip_buffer.seek(0)
        return send_file(
            zip_buffer,
            as_attachment=True,
            download_name=f"benchmark_{safe_name}_files.zip",
            mimetype="application/zip"
        )

    except Exception as e:
        return render_template('error.html',
                               message=f"Error saving benchmark: {str(e)}")



@app.route("/process", methods=["POST"])
def process():
    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ pre-flight â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if not XL:
        return render_template("error.html",
                               message="Install xlsxwriter or openpyxl.")

    dfn, mfn = request.form.get("data_fn"), request.form.get("map_fn")
    if not (dfn and mfn):
        return render_template("error.html", message="Session expired.")

    df = _read(os.path.join(UPLOAD, dfn))
    mdf = _read(os.path.join(UPLOAD, mfn))
    mdf["qt"] = mdf["question_type"].str.lower().str.strip()

    # build look-ups & groups
    qtypes = mdf.set_index("column_name")["qt"].to_dict()
    vmap, lorder, qtext, rankgrp, msgrp = _maps(mdf)

    rlabel = (mdf[mdf["qt"] == "ranking"]
              .drop_duplicates("column_name")
              .set_index("column_name")["value_label"].to_dict())

    mslabel = (
        mdf[
            mdf["qt"].isin(["multi_select", "boolean", "binary"])
            & mdf["group"].notna()
            ]
            .drop_duplicates("column_name")
            .set_index("column_name")["value_label"]
            .to_dict()
    )

    indeps = request.form.getlist("breakdowns")
    deps = request.form.getlist("analysis_vars")

    # Robust Validation
    warnings = []

    missing = sorted(set(indeps + deps) - set(df.columns))
    if missing:
        warnings.append(
            "Skipped â€“ column(s) not found in the data: " + ", ".join(missing)
        )
        indeps = [c for c in indeps if c in df.columns]
        deps = [c for c in deps if c in df.columns]

    bad_breaks = [b for b in indeps if qtypes.get(b, "categorical") not in CAT]
    if bad_breaks:
        warnings.append(
            "Skipped â€“ breakdown(s) not categorical: " + ", ".join(bad_breaks)
        )
        indeps = [b for b in indeps if b not in bad_breaks]

    # canonicalise categorical codes
    dfc = df.copy()
    for col in vmap:
        if col in dfc.columns:
            dfc[col] = dfc[col].apply(canon)

    rank_cols = set(sum(rankgrp.values(), []))
    ms_cols = set(sum(msgrp.values(), []))

    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ Excel setup â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    buf = io.BytesIO()
    writer = pd.ExcelWriter(buf, engine=XL)
    wb = writer.book

    if warnings:
        ws_warn = wb.add_worksheet("Warnings")
        writer.sheets["Warnings"] = ws_warn
        for i, msg in enumerate(warnings):
            ws_warn.write(i, 0, msg)
        if XL == "xlsxwriter":
            ws_warn.set_tab_color("#FFC000")
        else:
            ws_warn.sheet_properties.tabColor = "FFC000"

    f_head = wb.add_format({"bold": True, "bg_color": "#DCE6F1", "align": "center"})
    f_cnt = wb.add_format({"bold": True, "bg_color": "#4F81BD", "font_color": "white", "align": "center"})
    f_pct = wb.add_format({"bold": True, "bg_color": "#C0504D", "font_color": "white", "align": "center"})
    f_num = wb.add_format({"bold": True, "bg_color": "#9BBB59", "align": "center"})
    f_rank = wb.add_format({"bold": True, "bg_color": "#F79646", "font_color": "white", "align": "center"})

    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ main loop â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    for ind in ["_ALL_"] + indeps:
        sheet_name = "All Respondents" if ind == "_ALL_" else f"{ind} - {qtext.get(ind, ind)[:20]}"
        sheet_name = _safe_sheet(sheet_name)
        ws = wb.add_worksheet(sheet_name)
        writer.sheets[sheet_name] = ws

        if sheet_name == "All Respondents":
            if XL == "xlsxwriter":
                ws.set_tab_color("#0000FF")
            elif XL == "openpyxl":
                ws.sheet_properties.tabColor = "0000FF"

        r = 0
        ind_order = lorder.get(ind, []) if ind != "_ALL_" else None
        row_map = (lambda x: vmap[ind].get(x, x)) if ind != "_ALL_" else (lambda _: "Total")

        # â”€â”€â”€ 1. standard numeric / categorical â”€â”€â”€
        for dep in deps:
            if dep in rank_cols or dep in ms_cols:
                continue

            title = f"Dependent: {qtext.get(dep, dep)} ({dep})"
            if ind != "_ALL_":
                title += f"  vs  {qtext.get(ind, ind)} ({ind})"
            width = len(lorder.get(dep, [])) + 1 if dep in lorder else 5
            ws.merge_range(r, 0, r, width, title, f_head)
            r += 1

            if qtypes.get(dep, "categorical") in NUM:
                ser_all = pd.to_numeric(dfc[dep], errors="coerce")
                if ind == "_ALL_":
                    stats = {
                        "Mean": pd.Series([ser_all.mean().round(2)], index=["Total"]),
                        "Median": pd.Series([ser_all.median().round(2)], index=["Total"]),
                        "Mode": pd.Series(
                            [ser_all.mode().iat[0] if not ser_all.mode().empty else pd.NA],
                            index=["Total"])
                    }
                else:
                    grp = ser_all.groupby(dfc[ind])
                    stats = {
                        "Mean": grp.mean().round(2),
                        "Median": grp.median().round(2),
                        "Mode": grp.apply(lambda x: x.mode().iat[0] if not x.mode().empty else pd.NA)
                    }

                for name, ser in stats.items():
                    ws.write_string(r, 0, name, f_num);
                    r += 1
                    df_stat = ser.to_frame(name)
                    if ind != "_ALL_":
                        df_stat.index = df_stat.index.map(row_map)
                        df_stat = df_stat.reindex(ind_order)
                    df_stat.to_excel(excel_writer=writer,
                                     sheet_name=sheet_name,
                                     startrow=r)
                    ws.set_row(r, None, f_head);
                    r += len(df_stat.index) + 2

            else:
                ws.write_string(r, 0, "Counts", f_cnt);
                r += 1
                if ind == "_ALL_":
                    ct = dfc[dep].value_counts(dropna=False).to_frame("Total").T
                else:
                    ct = pd.crosstab(dfc[ind], dfc[dep])
                    ct.index = ct.index.map(row_map)
                    ct = ct.reindex(ind_order, fill_value=0)

                dep_ord = lorder.get(dep, list(ct.columns))
                ct.rename(columns=lambda x: vmap[dep].get(x, x), inplace=True)
                ct = ct.reindex(columns=dep_ord, fill_value=0)
                ct.to_excel(excel_writer=writer,
                            sheet_name=sheet_name,
                            startrow=r)
                ws.set_row(r, None, f_head);
                r += len(ct.index) + 1

                ws.write_string(r, 0, "% of total", f_pct);
                r += 1
                pct = (ct.div(ct.sum(axis=1), axis=0) * 100).round(1)
                pct.to_excel(excel_writer=writer,
                             sheet_name=sheet_name,
                             startrow=r)
                ws.set_row(r, None, f_head);
                r += len(pct.index) + 2

        # â”€â”€â”€ 2. ranking tables â”€â”€â”€
        selected_ranks = {g: cols for g, cols in rankgrp.items()
                          if any(c in deps for c in cols)}
        if selected_ranks:
            base_r = (dfc.shape[0] if ind == "_ALL_"
                      else dfc.groupby(ind).size()
                      .reindex(ind_order, fill_value=0)
                      .replace(0, 1))
            for grp, cols in selected_ranks.items():
                labels = [rlabel[c] for c in cols]
                ws.write_string(r, 0, f"Ranking: {grp}", f_head);
                r += 1

                for pos in (1, 2, 3):
                    ws.write_string(r, 0, f"Rank {pos}", f_rank);
                    r += 1
                    if ind == "_ALL_":
                        cnts = [(dfc[c] == canon(pos)).sum() for c in cols]
                        df_cnt = pd.DataFrame([cnts], index=["Total"], columns=labels)
                    else:
                        df_cnt = (dfc[cols] == canon(pos)).groupby(dfc[ind]).sum()
                        df_cnt.index = df_cnt.index.map(row_map)
                        df_cnt = df_cnt.reindex(ind_order, fill_value=0)
                        df_cnt.columns = labels
                    df_cnt.to_excel(excel_writer=writer,
                                    sheet_name=sheet_name,
                                    startrow=r)
                    ws.set_row(r, None, f_head);
                    r += len(df_cnt.index) + 1

                    ws.write_string(r, 0, "% of total", f_pct);
                    r += 1
                    if ind == "_ALL_":
                        pct_vals = [round(c / base_r * 100, 1) for c in cnts]
                        df_pct = pd.DataFrame([pct_vals],
                                              index=["Total"],
                                              columns=labels)
                    else:
                        df_pct = (df_cnt.div(base_r, axis=0) * 100).round(1)
                    df_pct.to_excel(excel_writer=writer,
                                    sheet_name=sheet_name,
                                    startrow=r)
                    ws.set_row(r, None, f_head);
                    r += len(df_pct.index) + 2

        # â”€â”€â”€ 3. multi-select tables â”€â”€â”€
        selected_ms = {g: cols for g, cols in msgrp.items()
                       if any(c in deps for c in cols)}
        if selected_ms:
            if ind == "_ALL_":
                base_ms = dfc.shape[0]
            else:
                base_ms = dfc.groupby(ind).size()
                base_ms.index = base_ms.index.map(row_map)
                base_ms = base_ms.reindex(ind_order, fill_value=0).replace(0, 1)

            for grp, cols_raw in selected_ms.items():
                cols = [c for c in dict.fromkeys(cols_raw) if c in dfc.columns]
                if not cols:
                    continue
                labels = [mslabel[c] for c in cols]

                ws.write_string(r, 0, f"Multi-select: {grp}", f_head);
                r += 1

                ws.write_string(r, 0, "Counts", f_cnt);
                r += 1
                if ind == "_ALL_":
                    cnts = [(pd.to_numeric(dfc[c], errors="coerce") == 1).sum()
                            for c in cols]
                    df_cnt = pd.DataFrame([cnts],
                                          index=["Total"],
                                          columns=labels)
                else:
                    bool_df = dfc[cols].apply(lambda s:
                                              pd.to_numeric(s, errors="coerce") == 1)
                    df_cnt = bool_df.groupby(dfc[ind]).sum()
                    df_cnt.index = df_cnt.index.map(row_map)
                    df_cnt = df_cnt.reindex(ind_order, fill_value=0)
                    df_cnt.columns = labels

                df_cnt.to_excel(excel_writer=writer,
                                sheet_name=sheet_name,
                                startrow=r)
                ws.set_row(r, None, f_head);
                r += len(df_cnt.index) + 1

                ws.write_string(r, 0, "% of total", f_pct);
                r += 1
                if ind == "_ALL_":
                    pct_vals = [round(c / base_ms * 100, 1) for c in cnts]
                    df_pct = pd.DataFrame([pct_vals],
                                          index=["Total"],
                                          columns=labels)
                else:
                    df_pct = (df_cnt.div(base_ms, axis=0) * 100).round(1)

                df_pct.to_excel(excel_writer=writer,
                                sheet_name=sheet_name,
                                startrow=r)
                ws.set_row(r, None, f_head);
                r += len(df_pct.index) + 2

    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ finish â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    writer.close()
    buf.seek(0)
    return send_file(
        buf,
        as_attachment=True,
        download_name="analysis_results.xlsx",
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


def open_browser():
    webbrowser.open_new("http://127.0.0.1:5000/")


if __name__ == '__main__':
    if not os.path.isdir(TPL):
        raise RuntimeError("templates/ folder missing")
    threading.Timer(1.0, open_browser).start()
    app.run(debug=False, port=5000)