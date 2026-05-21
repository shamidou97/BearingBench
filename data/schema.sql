-- ============================================================
-- BearingBench — MySQL Schema
-- XJTU-SY Rolling Element Bearing Dataset
-- Run: sudo mysql bearingbench < data/schema.sql
-- ============================================================

USE bearingbench;

-- ── Working Conditions ────────────────────────────────────────
CREATE TABLE IF NOT EXISTS working_conditions (
    id              INT AUTO_INCREMENT PRIMARY KEY,
    name            VARCHAR(10)  NOT NULL UNIQUE,  -- WC1, WC2, WC3
    rpm             INT          NOT NULL,
    load_kn         FLOAT        NOT NULL,
    severity_index  BIGINT       NOT NULL,          -- load_kn × rpm²
    description     VARCHAR(100),
    created_at      TIMESTAMP    DEFAULT CURRENT_TIMESTAMP
);

-- ── Bearings ──────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS bearings (
    id              INT AUTO_INCREMENT PRIMARY KEY,
    wc_id           INT          NOT NULL,
    name            VARCHAR(20)  NOT NULL UNIQUE,   -- Bearing1_1 etc.
    total_files     INT          NOT NULL,
    lifetime_min    INT          NOT NULL,           -- minutes = total_files
    fault_label     TINYINT      NOT NULL,           -- 0=Normal 1=Inner 2=Outer 3=Cage
    fault_type      VARCHAR(30)  NOT NULL,
    is_skipped      BOOLEAN      NOT NULL DEFAULT FALSE,
    skip_reason     VARCHAR(100),
    created_at      TIMESTAMP    DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (wc_id) REFERENCES working_conditions(id),
    INDEX idx_fault_label (fault_label),
    INDEX idx_wc          (wc_id)
);

-- ── Files (per-CSV metadata) ───────────────────────────────────
CREATE TABLE IF NOT EXISTS files (
    id              INT AUTO_INCREMENT PRIMARY KEY,
    bearing_id      INT          NOT NULL,
    file_number     INT          NOT NULL,           -- 1, 2, 3 ... N
    filename        VARCHAR(20)  NOT NULL,           -- 1.csv, 2.csv ...
    lifetime_pct    FLOAT        NOT NULL,           -- file_number / total_files × 100
    fault_label     TINYINT      NOT NULL,           -- 0=Normal, fault type for last 20%
    fault_state     VARCHAR(15)  NOT NULL,           -- Normal / Inner Race / Outer Race / Cage
    created_at      TIMESTAMP    DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (bearing_id) REFERENCES bearings(id),
    INDEX idx_bearing_file  (bearing_id, file_number),
    INDEX idx_fault_label   (fault_label)
);

-- ── Model Results ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS model_results (
    id                  INT AUTO_INCREMENT PRIMARY KEY,
    model_name          VARCHAR(50)  NOT NULL,
    input_shape         VARCHAR(50)  NOT NULL,
    params              INT,
    -- Overall metrics
    accuracy            FLOAT,
    macro_f1            FLOAT,
    -- Per-class metrics
    precision_normal    FLOAT,
    recall_normal       FLOAT,
    f1_normal           FLOAT,
    precision_inner     FLOAT,
    recall_inner        FLOAT,
    f1_inner            FLOAT,
    precision_outer     FLOAT,
    recall_outer        FLOAT,
    f1_outer            FLOAT,
    precision_cage      FLOAT,
    recall_cage         FLOAT,
    f1_cage             FLOAT,
    -- AUC per class
    auc_normal          FLOAT,
    auc_inner           FLOAT,
    auc_outer           FLOAT,
    auc_cage            FLOAT,
    -- Training info
    training_time_sec   FLOAT,
    epochs_trained      INT,
    batch_size          INT,
    learning_rate       FLOAT,
    model_path          VARCHAR(200),
    trained_at          TIMESTAMP    DEFAULT CURRENT_TIMESTAMP
);

-- ── Views ─────────────────────────────────────────────────────
CREATE OR REPLACE VIEW v_bearing_summary AS
SELECT
    wc.name                         AS wc_condition,
    wc.rpm,
    wc.load_kn,
    b.name                          AS bearing,
    b.total_files,
    b.lifetime_min                  AS lifetime_minutes,
    b.fault_type,
    b.is_skipped,
    COUNT(f.id)                     AS total_csv_records,
    SUM(f.fault_label = 0)          AS normal_files,
    SUM(f.fault_label > 0)          AS fault_files
FROM bearings b
JOIN working_conditions wc ON wc.id = b.wc_id
LEFT JOIN files f          ON f.bearing_id = b.id
GROUP BY b.id
ORDER BY wc.name, b.name;

CREATE OR REPLACE VIEW v_model_comparison AS
SELECT
    model_name,
    input_shape,
    params,
    ROUND(accuracy * 100, 1)        AS accuracy_pct,
    ROUND(macro_f1, 3)              AS macro_f1,
    ROUND(recall_cage, 3)           AS cage_recall,
    ROUND(auc_cage, 3)              AS cage_auc,
    ROUND(recall_inner, 3)          AS inner_recall,
    ROUND(auc_inner, 3)             AS inner_auc,
    training_time_sec,
    epochs_trained
FROM model_results
ORDER BY accuracy DESC;

CREATE OR REPLACE VIEW v_class_distribution AS
SELECT
    b.name                          AS bearing,
    wc.name                         AS wc_condition,
    b.fault_type,
    COUNT(f.id)                     AS total_files,
    SUM(f.fault_label = 0)          AS normal,
    SUM(f.fault_label = 1)          AS inner_race,
    SUM(f.fault_label = 2)          AS outer_race,
    SUM(f.fault_label = 3)          AS cage
FROM files f
JOIN bearings b            ON b.id = f.bearing_id
JOIN working_conditions wc ON wc.id = b.wc_id
GROUP BY b.id
ORDER BY wc.name, b.name;
