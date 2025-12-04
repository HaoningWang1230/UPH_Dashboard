import os
from datetime import datetime, date, timedelta
from typing import List, Dict, Tuple

import pandas as pd
import numpy as np
import streamlit as st
import altair as alt  # 用于画有数据点的折线图

# ========== 基础配置 ==========
st.set_page_config(
    page_title="效率监控看板",
    layout="wide",
)

DATA_DIR = "./data"
os.makedirs(DATA_DIR, exist_ok=True)

LABOR_STORE = os.path.join(DATA_DIR, "labor.parquet")
CLOCK_STORE = os.path.join(DATA_DIR, "clock.parquet")
MAP_STORE = os.path.join(DATA_DIR, "map.parquet")

# 六种任务类型
WORK_TYPES = [
    "出库拣选/普通单件",
    "出库拣选/爆品组合",
    "出库拣选/混合包裹",
    "出库复核/爆品组合",
    "出库复核/混合包裹",
    "出库复核/普通单件",
]

# 任务类型 -> 环节
TYPE_TO_SEG = {
    "出库拣选/普通单件": "拣选",
    "出库拣选/爆品组合": "拣选",
    "出库拣选/混合包裹": "拣选",
    "出库复核/普通单件": "散单打包",
    "出库复核/混合包裹": "散单打包",
    "出库复核/爆品组合": "秒杀打标",
}

# 打卡列 -> 任务类型名字
CLOCK_TYPE_MAP = {
    "上下班卡1": "上班",
    "上下班卡2": "下班",
    "午餐卡1": "开始午餐",
    "午餐卡2": "结束午餐",
}

CARD_COLS = ["上下班卡1", "上下班卡2", "午餐卡1", "午餐卡2"]


# ========== 工具函数：时间解析 & I/O ==========
def parse_datetime_series(s: pd.Series) -> pd.Series:
    """解析 12 小时制字符串到 datetime."""
    s_str = s.astype(str).str.strip()
    dt = pd.to_datetime(s_str, format="%m/%d/%Y %I:%M:%S %p", errors="coerce")
    mask = dt.isna()
    if mask.any():
        dt2 = pd.to_datetime(s_str[mask], errors="coerce")
        dt[mask] = dt2
    return dt


def combine_date_time(date_series: pd.Series, time_series: pd.Series) -> pd.Series:
    """把 考勤日期(date) + 时间字符串 拼成 datetime."""
    d = pd.to_datetime(date_series, errors="coerce").dt.date
    t_str = time_series.astype(str).str.strip()

    combined = []
    for d_val, t_val in zip(d, t_str):
        if pd.isna(d_val) or t_val in ["", "NaT", "nan", "NaN", "None"]:
            combined.append(pd.NaT)
        else:
            s = f"{d_val.strftime('%m/%d/%Y')} {t_val}"
            combined.append(pd.to_datetime(s, errors="coerce"))
    return pd.to_datetime(pd.Series(combined))


def read_parquet_if_exists(path: str) -> pd.DataFrame:
    if os.path.exists(path):
        return pd.read_parquet(path)
    return pd.DataFrame()


def append_and_dedup(
    old_df: pd.DataFrame,
    new_df: pd.DataFrame,
    subset_cols: List[str],
) -> pd.DataFrame:
    if old_df.empty:
        combined = new_df.copy()
    else:
        combined = pd.concat([old_df, new_df], ignore_index=True)
    combined = combined.drop_duplicates(subset=subset_cols)
    return combined


# ========== 1. 三类原始数据上传 & 标准化 ==========

@st.cache_data(show_spinner=False)
def load_history() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """从本地 parquet 读取历史数据，没有则返回空表。"""
    labor_hist = read_parquet_if_exists(LABOR_STORE)
    clock_hist = read_parquet_if_exists(CLOCK_STORE)
    map_hist = read_parquet_if_exists(MAP_STORE)
    return labor_hist, clock_hist, map_hist


def standardize_labor(upload: bytes) -> pd.DataFrame:
    """将出库劳动力原始 Excel 标准化为统一字段."""
    raw = pd.read_excel(upload)

    # 只保留需要字段并重命名
    labor_df = raw[["账号", "开始时间", "完成时间", "作业数量", "SKU数", "任务类型"]].copy()
    labor_df["StaffID"] = labor_df["账号"].astype(str)

    # 时间字段
    labor_df["StartTime"] = parse_datetime_series(labor_df["开始时间"])
    labor_df["EndTime"] = parse_datetime_series(labor_df["完成时间"])

    # 数量字段
    labor_df["Qty"] = pd.to_numeric(labor_df["作业数量"], errors="coerce").fillna(0)
    labor_df["SKUQty"] = pd.to_numeric(labor_df["SKU数"], errors="coerce")

    # 任务类型
    labor_df["TaskType"] = labor_df["任务类型"].astype(str)
    labor_df = labor_df[labor_df["TaskType"].isin(WORK_TYPES)].copy()

    # 剔除跨天任务
    same_day_mask = labor_df["StartTime"].dt.date == labor_df["EndTime"].dt.date
    labor_df = labor_df[same_day_mask].copy()

    # 日期 & 环节
    labor_df["Date"] = labor_df["StartTime"].dt.date
    labor_df["Segment"] = labor_df["TaskType"].map(TYPE_TO_SEG)

    return labor_df[
        [
            "StaffID",
            "StartTime",
            "EndTime",
            "Qty",
            "SKUQty",
            "TaskType",
            "Segment",
            "Date",
        ]
    ]


def standardize_map(upload: bytes) -> pd.DataFrame:
    """标准化姓名–账号映射表."""
    m = pd.read_excel(upload)

    # 支持中/英列名
    if "账号" in m.columns and "姓名" in m.columns:
        m = m.rename(columns={"账号": "StaffID", "姓名": "StaffName"})
    elif "Account" in m.columns and "Name" in m.columns:
        m = m.rename(columns={"Account": "StaffID", "Name": "StaffName"})
    else:
        st.error("姓名–账号映射表必须包含 [账号, 姓名] 或 [Account, Name] 两列")
        st.stop()

    if "职位" in m.columns:
        m = m.rename(columns={"职位": "Position"})
    else:
        m["Position"] = None

    m["StaffID"] = m["StaffID"].astype(str)
    return m[["StaffID", "StaffName", "Position"]]


def standardize_clock(upload: bytes, map_df: pd.DataFrame) -> pd.DataFrame:
    """标准化考勤表，并计算“全员工时”字段."""
    raw = pd.read_excel(upload)

    clock_df = raw[["姓名", "考勤日期"] + CARD_COLS].copy()
    if "职位" in raw.columns:
        clock_df["Position"] = raw["职位"]
    else:
        clock_df["Position"] = None

    # ===== 姓名 -> StaffID；映射失败就用原姓名，不要 NaN =====
    name_to_id = dict(zip(map_df["StaffName"], map_df["StaffID"]))
    clock_df["StaffID"] = clock_df["姓名"].map(name_to_id)
    mask_missing = clock_df["StaffID"].isna()
    clock_df.loc[mask_missing, "StaffID"] = clock_df.loc[mask_missing, "姓名"]
    clock_df["StaffID"] = clock_df["StaffID"].astype(str)
    # ====================================================

    # 日期
    clock_df["Date"] = pd.to_datetime(clock_df["考勤日期"], errors="coerce").dt.date

    # 四张卡转 timestamp
    for col in CARD_COLS:
        clock_df[col] = combine_date_time(clock_df["Date"], clock_df[col])

    # ---- 计算该天总工作时长（用于 Overall UPH）----
    # 优先使用 TK 里自带的「总工时」列（通常已扣除午餐）
    if "总工时" in raw.columns:
        work_hours = pd.to_numeric(raw["总工时"], errors="coerce")
    else:
        # 没有「总工时」列时，才用打卡时间计算
        def calc_hours(row):
            s1, s2, l1, l2 = row["上下班卡1"], row["上下班卡2"], row["午餐卡1"], row["午餐卡2"]
            if pd.isna(s1) or pd.isna(s2):
                return 0.0
            total = (s2 - s1).total_seconds() / 3600.0
            lunch = 0.0
            if pd.notna(l1) and pd.notna(l2) and l2 > l1:
                lunch = (l2 - l1).total_seconds() / 3600.0
            return max(total - lunch, 0.0)

        work_hours = clock_df.apply(calc_hours, axis=1)

    clock_df["WorkHours_AllStaff"] = work_hours.fillna(0.0)

    return clock_df[
        [
            "StaffID",
            "姓名",
            "Position",
            "Date",
            "WorkHours_AllStaff",
        ]
        + CARD_COLS
    ]


# ========== 2. 侧边栏：三类上传 + 清除/替换 ==========

# 先读历史
labor_hist, clock_hist, map_hist = load_history()

st.sidebar.title("数据上传")

# ---- 账号信息 ----
map_files = st.sidebar.file_uploader(
    "请上传账号信息（姓名–账号映射表，可多次上传）",
    type=["xls", "xlsx"],
    accept_multiple_files=True,
    key="map_files",
)
if map_files:
    for f in map_files:
        map_new = standardize_map(f)
        map_hist = append_and_dedup(map_hist, map_new, subset_cols=["StaffID"])
    map_hist.to_parquet(MAP_STORE)
    load_history.clear()
    st.sidebar.success(f"账号信息已更新，当前共 {len(map_hist)} 名员工")

# ---- 劳动力数据 ----
st.sidebar.markdown("---")
st.sidebar.subheader("劳动力数据")

replace_labor = st.sidebar.checkbox("本次上传视为【替换】这些日期的劳动力数据", value=False)

labor_files = st.sidebar.file_uploader(
    "请上传劳动力数据（可多文件，自动累积去重）",
    type=["xls", "xlsx"],
    accept_multiple_files=True,
    key="labor_files",
)

if labor_files:
    for f in labor_files:
        labor_new = standardize_labor(f)
        if replace_labor and not labor_hist.empty:
            dates_to_replace = labor_new["Date"].unique()
            labor_hist = labor_hist[~labor_hist["Date"].isin(dates_to_replace)]
        labor_hist = append_and_dedup(
            labor_hist,
            labor_new,
            subset_cols=["StaffID", "StartTime", "EndTime", "TaskType", "Qty"],
        )
    labor_hist.to_parquet(LABOR_STORE)
    load_history.clear()
    st.sidebar.success(f"劳动力数据已累积：{len(labor_hist)} 条记录")


# ---- 考勤数据 ----
st.sidebar.markdown("---")
st.sidebar.subheader("考勤数据")

replace_clock = st.sidebar.checkbox("本次上传视为【替换】这些日期的考勤数据", value=False)

clock_files = st.sidebar.file_uploader(
    "请上传考勤数据（可多文件，自动累积去重）",
    type=["xls", "xlsx"],
    accept_multiple_files=True,
    key="clock_files",
)

if clock_files:
    if map_hist.empty:
        st.sidebar.error("请先上传账号信息（姓名–账号映射表），否则无法匹配考勤。")
    else:
        for f in clock_files:
            clock_new = standardize_clock(f, map_hist)
            if replace_clock and not clock_hist.empty:
                dates_to_replace = clock_new["Date"].unique()
                clock_hist = clock_hist[~clock_hist["Date"].isin(dates_to_replace)]
            # 去重时包含姓名，避免不同姓名但 StaffID 一样时互相覆盖
            clock_hist = append_and_dedup(
                clock_hist,
                clock_new,
                subset_cols=["StaffID", "Date", "姓名"],
            )
        clock_hist.to_parquet(CLOCK_STORE)
        load_history.clear()
        st.sidebar.success(f"考勤数据已累积：{len(clock_hist)} 条记录")


# ---- 清除 / 单日替换工具 ----
st.sidebar.markdown("---")
st.sidebar.subheader("数据清理 / 单日删除")

with st.sidebar.expander("删除某一天的数据"):
    data_type = st.radio("选择要删除的数据表", ["劳动力数据", "考勤数据"], horizontal=True)
    del_date = st.date_input("选择要删除的日期", value=date.today())

    if st.button("删除该日期数据", use_container_width=True):
        if data_type == "劳动力数据":
            if labor_hist.empty:
                st.warning("当前没有劳动力历史数据。")
            else:
                before = len(labor_hist)
                labor_hist = labor_hist[labor_hist["Date"] != del_date]
                removed = before - len(labor_hist)
                labor_hist.to_parquet(LABOR_STORE)
                load_history.clear()
                st.success(f"已从劳动力数据中删除 {removed} 条记录。")
        else:
            if clock_hist.empty:
                st.warning("当前没有考勤历史数据。")
            else:
                before = len(clock_hist)
                clock_hist = clock_hist[clock_hist["Date"] != del_date]
                removed = before - len(clock_hist)
                clock_hist.to_parquet(CLOCK_STORE)
                load_history.clear()
                st.success(f"已从考勤数据中删除 {removed} 条记录。")


# ========== 3. ETL 主逻辑（基于你给的 Python + Overall UPH） ==========
def run_etl(
    labor_df: pd.DataFrame,
    clock_df: pd.DataFrame,
    map_df: pd.DataFrame,
) -> Dict[str, pd.DataFrame]:
    """返回四张核心表：daily_overall_uph, segment_uph, tasktype_uph, staff_daily_uph"""

    # --- 准备映射 ---
    account_to_name = dict(zip(map_df["StaffID"], map_df["StaffName"]))

    # --- 处理劳动力数据 ---
    labor_df = labor_df.copy()
    labor_df["StaffID"] = labor_df["StaffID"].astype(str)
    labor_df["Date"] = labor_df["StartTime"].dt.date
    labor_df["Segment"] = labor_df["TaskType"].map(TYPE_TO_SEG)
    labor_df["StaffName"] = labor_df["StaffID"].map(account_to_name)
    # 映射不到的，用 StaffID 顶上，避免 NaN
    mask_sn = labor_df["StaffName"].isna()
    labor_df.loc[mask_sn, "StaffName"] = labor_df.loc[mask_sn, "StaffID"]

    # --- 处理考勤数据（含打卡展开） ---
    clock_raw = clock_df.copy()
    clock_raw["StaffName"] = clock_raw["姓名"]

    # 生成“午餐时间表”
    lunch_df = clock_raw[["StaffID", "Date", "午餐卡1", "午餐卡2"]].copy()

    # 展开打卡为长表：每条打卡当作一条“0件量任务”
    clock_long = clock_raw.melt(
    id_vars=["StaffID", "StaffName", "Date"],
    value_vars=CARD_COLS,
    var_name="打卡列",
    value_name="打卡时间",
)

    clock_long = clock_long.dropna(subset=["打卡时间"]).copy()
    clock_long["TaskType"] = clock_long["打卡列"].map(CLOCK_TYPE_MAP)
    clock_long["StartTime"] = clock_long["打卡时间"]
    clock_long["EndTime"] = clock_long["打卡时间"]
    clock_long["Qty"] = 0.0
    clock_long["Segment"] = None

    clock_std = clock_long[
        ["StaffID", "StaffName", "Date", "StartTime", "EndTime", "TaskType", "Qty", "Segment"]
    ].copy()

    # --- 午餐切段 ---
    labor_for_split = labor_df[
        ["StaffID", "StaffName", "Date", "StartTime", "EndTime", "Qty", "TaskType", "Segment"]
    ].copy()
    labor_for_split = labor_for_split.merge(
        lunch_df,
        on=["StaffID", "Date"],
        how="left",
    )

    def split_task_by_lunch(row):
        start = row["StartTime"]
        end = row["EndTime"]
        qty = row["Qty"]
        ttype = row["TaskType"]
        seg = row["Segment"]
        acc = row["StaffID"]
        name = row["StaffName"]
        d = row["Date"]
        l1 = row["午餐卡1"]
        l2 = row["午餐卡2"]

        if (
            pd.isna(start)
            or pd.isna(end)
            or pd.isna(d)
            or pd.isna(l1)
            or pd.isna(l2)
            or not (start < end < (start + pd.Timedelta(days=1)))
        ):
            return [
                dict(
                    StaffID=acc,
                    StaffName=name,
                    Date=d,
                    StartTime=start,
                    EndTime=end,
                    Qty=qty,
                    TaskType=ttype,
                    Segment=seg,
                )
            ]

        if not (l1 < l2):
            return [
                dict(
                    StaffID=acc,
                    StaffName=name,
                    Date=d,
                    StartTime=start,
                    EndTime=end,
                    Qty=qty,
                    TaskType=ttype,
                    Segment=seg,
                )
            ]

        if end <= l1 or start >= l2:
            return [
                dict(
                    StaffID=acc,
                    StaffName=name,
                    Date=d,
                    StartTime=start,
                    EndTime=end,
                    Qty=qty,
                    TaskType=ttype,
                    Segment=seg,
                )
            ]

        segs = []

        pre_start = start
        pre_end = min(end, l1)
        if pre_end > pre_start:
            segs.append(("pre", pre_start, pre_end))

        post_start = max(start, l2)
        post_end = end
        if post_end > post_start:
            segs.append(("post", post_start, post_end))

        if not segs:
            return [
                dict(
                    StaffID=acc,
                    StaffName=name,
                    Date=d,
                    StartTime=start,
                    EndTime=end,
                    Qty=qty,
                    TaskType=ttype,
                    Segment=seg,
                )
            ]

        durations = [(e - s).total_seconds() for _, s, e in segs]
        total_work_seconds = sum(durations)
        new_rows = []
        allocated_qty = 0.0

        for idx, (_, s, e) in enumerate(segs):
            dur = (e - s).total_seconds()
            if idx < len(segs) - 1:
                q_part = qty * dur / total_work_seconds
                allocated_qty += q_part
            else:
                q_part = qty - allocated_qty
            new_rows.append(
                dict(
                    StaffID=acc,
                    StaffName=name,
                    Date=d,
                    StartTime=s,
                    EndTime=e,
                    Qty=q_part,
                    TaskType=ttype,
                    Segment=seg,
                )
            )
        return new_rows

    split_rows = []
    for _, r in labor_for_split.iterrows():
        split_rows.extend(split_task_by_lunch(r))
    labor_splitted = pd.DataFrame(split_rows)

    # --- 拼时间轴 ---
    labor_splitted["Date"] = pd.to_datetime(labor_splitted["Date"]).dt.date
    clock_std["Date"] = pd.to_datetime(clock_std["Date"]).dt.date

    timeline_df = pd.concat([labor_splitted, clock_std], ignore_index=True)
    timeline_df = timeline_df.sort_values(
        by=["StaffID", "Date", "StartTime"]
    ).reset_index(drop=True)

    # --- 每人每天按任务类型算工时&件量 ---
    def compute_daily_task_hours_and_qty(sub: pd.DataFrame):
        sub = sub.sort_values("StartTime").reset_index(drop=True).copy()
        hours = {t: 0.0 for t in WORK_TYPES}
        qty = {t: 0.0 for t in WORK_TYPES}
        n = len(sub)
        i = 0
        while i < n:
            ttype = sub.loc[i, "TaskType"]
            if ttype not in WORK_TYPES:
                i += 1
                continue
            j = i
            while j + 1 < n and sub.loc[j + 1, "TaskType"] == ttype:
                j += 1
            seg_start = sub.loc[i, "StartTime"]
            if i - 1 >= 0:
                prev_type = sub.loc[i - 1, "TaskType"]
                if prev_type in ["上班", "结束午餐"]:
                    seg_start = sub.loc[i - 1, "StartTime"]
            if j + 1 < n:
                seg_end = sub.loc[j + 1, "StartTime"]
            else:
                seg_end = sub.loc[j, "EndTime"]
            if pd.notna(seg_start) and pd.notna(seg_end) and seg_end > seg_start:
                delta_h = (seg_end - seg_start).total_seconds() / 3600.0
                hours[ttype] += delta_h
                block_qty = sub.loc[i:j, "Qty"].sum()
                qty[ttype] += block_qty
            i = j + 1
        return hours, qty

    records = []
    for (acc, d), sub in timeline_df.groupby(["StaffID", "Date"], sort=False):
        name = sub["StaffName"].dropna().iloc[0] if sub["StaffName"].notna().any() else acc
        h_dict, q_dict = compute_daily_task_hours_and_qty(sub)
        for ttype in WORK_TYPES:
            h = h_dict.get(ttype, 0.0)
            q = q_dict.get(ttype, 0.0)
            if h <= 0 and q == 0:
                continue
            records.append(
                dict(
                    Date=d,
                    StaffID=acc,
                    StaffName=name,
                    TaskType=ttype,
                    Segment=TYPE_TO_SEG.get(ttype),
                    TotalQty=q,
                    TaskHours=h,
                )
            )
    per_person_task = pd.DataFrame(records)
    if per_person_task.empty:
        return {
            "daily_overall_uph": pd.DataFrame(),
            "segment_uph": pd.DataFrame(),
            "tasktype_uph": pd.DataFrame(),
            "staff_daily_uph": pd.DataFrame(),
            "per_person_task": pd.DataFrame(),
        }

    per_person_task["UPH_task"] = (
        per_person_task["TotalQty"] / per_person_task["TaskHours"]
    ).replace([np.inf, -np.inf], np.nan)
    per_person_task["UPH_task"] = per_person_task["UPH_task"].round(2)

    # --- 环节 UPH ---
    seg_daily = (
        per_person_task.dropna(subset=["Segment"])
        .groupby(["Date", "Segment"], as_index=False)
        .agg(TotalQty=("TotalQty", "sum"), Hours=("TaskHours", "sum"))
    )
    seg_daily["UPH"] = (seg_daily["TotalQty"] / seg_daily["Hours"]).replace(
        [np.inf, -np.inf], np.nan
    )
    seg_daily["UPH"] = seg_daily["UPH"].round(2)

    # --- 任务类型 UPH ---
    task_daily = (
        per_person_task.groupby(["Date", "TaskType"], as_index=False)
        .agg(TotalQty=("TotalQty", "sum"), Hours=("TaskHours", "sum"))
    )
    task_daily["UPH"] = (task_daily["TotalQty"] / task_daily["Hours"]).replace(
        [np.inf, -np.inf], np.nan
    )
    task_daily["UPH"] = task_daily["UPH"].round(2)

    # --- 员工日 UPH（总量） ---
    staff_daily = (
        per_person_task.groupby(["Date", "StaffID", "StaffName"], as_index=False)
        .agg(TotalQty=("TotalQty", "sum"), Hours=("TaskHours", "sum"))
    )
    staff_daily["TotalUPH"] = (staff_daily["TotalQty"] / staff_daily["Hours"]).replace(
        [np.inf, -np.inf], np.nan
    )
    staff_daily["TotalUPH"] = staff_daily["TotalUPH"].round(2)
    staff_daily["Rank"] = staff_daily.groupby("Date")["TotalUPH"].rank(
        ascending=False, method="min"
    )

    # --- Overall UPH：复核三种类型件量 / 全员总工时 ---
    review_types = [
        "出库复核/爆品组合",
        "出库复核/混合包裹",
        "出库复核/普通单件",
    ]

    # 1）当天复核三种类型作业数量之和 = 总件量（散单打包 + 秒杀打标）
    daily_qty = (
        per_person_task[per_person_task["TaskType"].isin(review_types)]
        .groupby("Date", as_index=False)
        .agg(TotalQtyReview=("TotalQty", "sum"))
    )

    # 2）当天所有人的总工时之和
    clock_for_hours = clock_df.copy()
    clock_for_hours["Date"] = pd.to_datetime(clock_for_hours["Date"]).dt.date

    workhours_daily = (
        clock_for_hours
        .groupby("Date", as_index=False)
        .agg(TotalHoursAll=("WorkHours_AllStaff", "sum"))
    )

    # 3）汇总成按天 Overall UPH
    daily_overall = workhours_daily.merge(daily_qty, on="Date", how="left")
    daily_overall["TotalQtyReview"] = daily_overall["TotalQtyReview"].fillna(0.0)

    daily_overall["UPH"] = (
        daily_overall["TotalQtyReview"] / daily_overall["TotalHoursAll"]
    ).replace([np.inf, -np.inf], np.nan)
    daily_overall["UPH"] = daily_overall["UPH"].round(2)

    # 标准字段名：total_qty = 三种复核总件量
    daily_overall_uph = daily_overall.rename(
        columns={
            "Date": "date",
            "TotalQtyReview": "total_qty",
            "TotalHoursAll": "total_hours",
        }
    )
    segment_uph = seg_daily.rename(
        columns={"Date": "date", "Segment": "segment", "TotalQty": "qty", "Hours": "hours"}
    )
    tasktype_uph = task_daily.rename(
        columns={"Date": "date", "TaskType": "tasktype", "TotalQty": "qty", "Hours": "hours"}
    )
    staff_daily_uph = staff_daily.rename(
        columns={
            "Date": "date",
            "StaffID": "staffid",
            "StaffName": "staff_name",
            "TotalQty": "total_qty",
            "Hours": "total_hours",
            "TotalUPH": "total_uph",
            "Rank": "rank",
        }
    )

    return {
        "daily_overall_uph": daily_overall_uph,
        "segment_uph": segment_uph,
        "tasktype_uph": tasktype_uph,
        "staff_daily_uph": staff_daily_uph,
        "per_person_task": per_person_task,
    }


# ========== 4. UI：Ant Design 风格的 Streamlit 看板 ==========

ANTD_CSS = """
<style>
.main {
    background-color: #f5f7fa;
}
.ant-card {
    background: #ffffff;
    border-radius: 12px;
    padding: 16px 20px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    border: 1px solid #f0f0f0;
}
.kpi-title {
    font-size: 13px;
    color: #8c8c8c;
    margin-bottom: 4px;
}
.kpi-value {
    font-size: 26px;
    font-weight: 600;
    color: #262626;
}
.kpi-sub {
    font-size: 12px;
    color: #8c8c8c;
}
.kpi-up {
    color: #52c41a;
    font-size: 12px;
    font-weight: 500;
}
.kpi-down {
    color: #f5222d;
    font-size: 12px;
    font-weight: 500;
}
.row-top3 {
    background: #f6ffed;
    border-left: 4px solid #52c41a;
}
.row-bottom3 {
    background: #fff1f0;
    border-left: 4px solid #f5222d;
}
</style>
"""

st.markdown(ANTD_CSS, unsafe_allow_html=True)

st.title("TK效率监控看板")

# 如果还没上传够三类数据，直接提示
if labor_hist.empty or clock_hist.empty or map_hist.empty:
    st.warning("请在左侧上传：姓名–账号映射表、出库劳动力数据、考勤数据。")
    st.stop()

# ---------- 4.1 运行 ETL ----------
with st.spinner("正在计算 UPH 指标..."):
    etl_result = run_etl(labor_hist, clock_hist, map_hist)

daily_overall = etl_result["daily_overall_uph"]
segment_uph = etl_result["segment_uph"]
tasktype_uph = etl_result["tasktype_uph"]
staff_daily_uph = etl_result["staff_daily_uph"]
per_person_task = etl_result["per_person_task"]

if daily_overall.empty:
    st.warning("当前数据无法计算 UPH（可能没有有效任务记录）。")
    st.stop()

# ---------- 4.2 全局筛选器 ----------
st.sidebar.markdown("## 筛选条件")

min_date = min(
    daily_overall["date"].min(),
    segment_uph["date"].min(),
    tasktype_uph["date"].min(),
)
max_date = max(
    daily_overall["date"].max(),
    segment_uph["date"].max(),
    tasktype_uph["date"].max(),
)

default_start = max_date - timedelta(days=6)

date_range = st.sidebar.date_input(
    "日期范围",
    value=(default_start, max_date),
    min_value=min_date,
    max_value=max_date,
)

if isinstance(date_range, (list, tuple)):
    start_date, end_date = date_range
else:
    start_date = date_range
    end_date = date_range

# 所有可选项先算出来
seg_options = sorted(segment_uph["segment"].dropna().unique().tolist())
type_options = sorted(tasktype_uph["tasktype"].dropna().unique().tolist())
staff_options = (
    staff_daily_uph["staff_name"]
    .dropna()
    .drop_duplicates()
    .sort_values()
    .tolist()
)

# 高级筛选放在折叠里，默认收起，避免侧边栏一长串
with st.sidebar.expander("更多筛选（环节 / 任务类型 / 员工）", expanded=False):
    sel_segments = st.multiselect("环节 Segment", seg_options, default=seg_options)
    sel_types = st.multiselect("任务类型 TaskType", type_options, default=type_options)
    sel_staffs = st.multiselect("员工 Staff", staff_options, default=staff_options)

# 过滤
mask_date = (daily_overall["date"] >= start_date) & (daily_overall["date"] <= end_date)
daily_overall_f = daily_overall[mask_date].copy()

segment_uph_f = segment_uph[
    (segment_uph["date"] >= start_date)
    & (segment_uph["date"] <= end_date)
    & (segment_uph["segment"].isin(sel_segments))
].copy()

tasktype_uph_f = tasktype_uph[
    (tasktype_uph["date"] >= start_date)
    & (tasktype_uph["date"] <= end_date)
    & (tasktype_uph["tasktype"].isin(sel_types))
].copy()

staff_daily_uph_f = staff_daily_uph[
    (staff_daily_uph["date"] >= start_date)
    & (staff_daily_uph["date"] <= end_date)
    & (staff_daily_uph["staff_name"].isin(sel_staffs))
].copy()

# 供多处使用的按日期过滤后的 per_person_task
per_person_task_f = per_person_task[
    (per_person_task["Date"] >= start_date)
    & (per_person_task["Date"] <= end_date)
].copy()

# 全局选中的员工（用于排行榜 <-> 个人详情页联动）
if "selected_staff" not in st.session_state:
    st.session_state["selected_staff"] = None


# ---------- 4.3 顶部 KPI 卡片 ----------
def render_kpi_card(title: str, today_value: float, yesterday_value: float):
    diff = None
    cls = ""
    arrow = ""
    if pd.notna(today_value) and pd.notna(yesterday_value) and yesterday_value != 0:
        diff = (today_value - yesterday_value) / yesterday_value * 100
        if diff > 0:
            cls = "kpi-up"
            arrow = "↑"
        elif diff < 0:
            cls = "kpi-down"
            arrow = "↓"

    if pd.notna(today_value):
        val_str = f"{today_value:.2f}"
    else:
        val_str = "0.00"

    # 时间说明
    if isinstance(today, (datetime, date)):
        today_str = today.strftime("%Y-%m-%d")
    else:
        today_str = str(today)

    if isinstance(yesterday, (datetime, date)):
        y_str = yesterday.strftime("%Y-%m-%d")
    else:
        y_str = str(yesterday)

    html = '<div class="ant-card">'
    # 标注这是哪一天的数据
    html += f'<div class="kpi-title">{title}（最近一天：{today_str}）</div>'
    html += f'<div class="kpi-value">{val_str} <span class="kpi-sub">UPH</span></div>'
    # 标注清楚和哪一天对比
    if diff is not None:
        html += f'<div class="{cls}">{arrow} {diff:.1f}% vs 前一日（{y_str}）</div>'
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


today = daily_overall_f["date"].max()
yesterday = today - timedelta(days=1)

# 说明文字：卡片是“最近一天 vs 前一日”
st.caption(
    f"顶部卡片说明：数值为所选日期范围内【最近一天】（{today}）的当日 UPH，"
    f"下方百分比为相对于前一日（{yesterday}）的变化。"
)


def get_uph(df: pd.DataFrame, date_col: str, uph_col: str, **filters):
    sub = df[df[date_col] == today]
    for k, v in filters.items():
        sub = sub[sub[k] == v]
    today_val = sub[uph_col].mean() if not sub.empty else np.nan

    sub_y = df[df[date_col] == yesterday]
    for k, v in filters.items():
        sub_y = sub_y[sub_y[k] == v]
    y_val = sub_y[uph_col].mean() if not sub_y.empty else np.nan
    return today_val, y_val


col1, col2, col3, col4 = st.columns(4)

with col1:
    t, y = get_uph(daily_overall_f, "date", "UPH")
    render_kpi_card("Overall UPH", t, y)

with col2:
    t, y = get_uph(segment_uph_f, "date", "UPH", segment="拣选")
    render_kpi_card("拣选 UPH", t, y)

with col3:
    t, y = get_uph(segment_uph_f, "date", "UPH", segment="散单打包")
    render_kpi_card("散单打包 UPH", t, y)

with col4:
    t, y = get_uph(segment_uph_f, "date", "UPH", segment="秒杀打标")
    render_kpi_card("秒杀打标 UPH", t, y)


# ---------- 4.4 Tabs：多个页面 ----------
tab_overall, tab_seg, tab_task, tab_staff, tab_detail = st.tabs(
    ["总览", "环节", "任务类型", "排名", "个人详情"]
)

# ===== 总览页 =====
with tab_overall:
    st.subheader("Overall UPH 日趋势")

    overall_chart_df = daily_overall_f.sort_values("date").copy()
    # 关键：加一个字符串日期列，用作 X 轴，避免时区问题 + 日期短一点
    overall_chart_df["date_ts"] = pd.to_datetime(overall_chart_df["date"])
    overall_chart_df["date_str"] = overall_chart_df["date_ts"].dt.strftime("%m-%d")

    if overall_chart_df.empty:
        st.info("当前筛选日期内没有 Overall UPH 数据。")
    else:
        # 折线 + 数据点（按天）
        chart = (
            alt.Chart(overall_chart_df)
            .mark_line(point=True)
            .encode(
                x=alt.X(
                    "date_str:N",
                    title="日期",
                    axis=alt.Axis(labelAngle=0),  # 横着显示
                ),
                y=alt.Y("UPH:Q", title="Overall UPH"),
                tooltip=[
                    alt.Tooltip("date_ts:T", title="日期"),
                    alt.Tooltip("total_qty:Q", title="件量（散单打包+秒杀打标）", format=".0f"),
                    alt.Tooltip("total_hours:Q", title="总工时(h，所有员工)", format=".2f"),
                    alt.Tooltip("UPH:Q", title="Overall UPH", format=".2f"),
                ],
            )
            .properties(height=260)
        )
        st.altair_chart(chart, use_container_width=True)

        st.markdown("### 每日明细")

        # 明细表：每天的件量 / 工时 / UPH
        detail_df = overall_chart_df[["date", "total_qty", "total_hours", "UPH"]].copy()
        detail_df = detail_df.rename(
            columns={
                "date": "日期",
                # 件量 = 三种复核任务（散单打包两种 + 秒杀打标）的总和
                "total_qty": "件量（散单打包+秒杀打标）",
                # 工时 = 当天所有员工的总工时（来自考勤表 WorkHours_AllStaff 求和）
                "total_hours": "总工时(h，所有员工)",
                "UPH": "Overall UPH",
            }
        )

        st.dataframe(detail_df, use_container_width=True)

        # 导出 CSV
        csv = detail_df.to_csv(index=False).encode("utf-8-sig")
        file_name = f"overall_detail_{start_date}_{end_date}.csv"
        st.download_button(
            "导出明细（CSV）",
            data=csv,
            file_name=file_name,
            mime="text/csv",
            use_container_width=True,
        )


# ===== 环节页 =====
with tab_seg:
    st.subheader("环节 UPH 日趋势")

    seg_trend_df = segment_uph_f.sort_values(["date", "segment"]).copy()
    seg_trend_df["date_ts"] = pd.to_datetime(seg_trend_df["date"])
    seg_trend_df["date_str"] = seg_trend_df["date_ts"].dt.strftime("%m-%d")

    if seg_trend_df.empty:
        st.info("当前筛选条件下没有环节数据。")
    else:
        # 在页面内再选一遍要展示的环节，图和明细都一起筛
        seg_all_in_tab = sorted(seg_trend_df["segment"].dropna().unique().tolist())
        sel_seg_tab = st.multiselect(
            "选择要展示的环节",
            seg_all_in_tab,
            default=seg_all_in_tab,
        )
        if sel_seg_tab:
            seg_trend_df = seg_trend_df[seg_trend_df["segment"].isin(sel_seg_tab)]

        # 多折线趋势图
        chart = (
            alt.Chart(seg_trend_df)
            .mark_line(point=True)
            .encode(
                x=alt.X(
                    "date_str:N",
                    title="日期",
                    axis=alt.Axis(labelAngle=0),
                ),
                y=alt.Y("UPH:Q", title="环节 UPH"),
                color=alt.Color("segment:N", title="环节"),
                tooltip=[
                    alt.Tooltip("date_ts:T", title="日期"),
                    alt.Tooltip("segment:N", title="环节"),
                    alt.Tooltip("qty:Q", title="件量", format=".0f"),
                    alt.Tooltip("hours:Q", title="工时(h)", format=".2f"),
                    alt.Tooltip("UPH:Q", title="UPH", format=".2f"),
                ],
            )
            .properties(height=280)
        )
        st.altair_chart(chart, use_container_width=True)

        st.markdown("### 每日环节明细")

        seg_detail_df = (
            seg_trend_df[["date", "segment", "qty", "hours", "UPH"]]
            .sort_values(["date", "segment"])
            .copy()
        )
        seg_detail_df = seg_detail_df.rename(
            columns={
                "date": "日期",
                "segment": "环节",
                "qty": "件量",
                "hours": "工时(h)",
                "UPH": "UPH",
            }
        )
        st.dataframe(seg_detail_df, use_container_width=True)

        seg_csv = seg_detail_df.to_csv(index=False).encode("utf-8-sig")
        seg_file_name = f"segment_detail_{start_date}_{end_date}.csv"
        st.download_button(
            "导出明细（CSV）",
            data=seg_csv,
            file_name=seg_file_name,
            mime="text/csv",
            use_container_width=True,
        )


# ===== 任务类型页 =====
with tab_task:
    st.subheader("任务类型 UPH 日趋势")

    task_trend_df = tasktype_uph_f.sort_values(["date", "tasktype"]).copy()
    task_trend_df["date_ts"] = pd.to_datetime(task_trend_df["date"])
    task_trend_df["date_str"] = task_trend_df["date_ts"].dt.strftime("%m-%d")

    if task_trend_df.empty:
        st.info("当前筛选条件下没有任务类型数据。")
    else:
        # 在页面内再选一遍要展示的任务类型，图和明细都一起筛
        task_all_in_tab = sorted(task_trend_df["tasktype"].dropna().unique().tolist())
        sel_task_tab = st.multiselect(
            "选择要展示的任务类型",
            task_all_in_tab,
            default=task_all_in_tab,
        )
        if sel_task_tab:
            task_trend_df = task_trend_df[task_trend_df["tasktype"].isin(sel_task_tab)]

        chart = (
            alt.Chart(task_trend_df)
            .mark_line(point=True)
            .encode(
                x=alt.X(
                    "date_str:N",
                    title="日期",
                    axis=alt.Axis(labelAngle=0),
                ),
                y=alt.Y("UPH:Q", title="任务类型 UPH"),
                color=alt.Color("tasktype:N", title="任务类型"),
                tooltip=[
                    alt.Tooltip("date_ts:T", title="日期"),
                    alt.Tooltip("tasktype:N", title="任务类型"),
                    alt.Tooltip("qty:Q", title="件量", format=".0f"),
                    alt.Tooltip("hours:Q", title="工时(h)", format=".2f"),
                    alt.Tooltip("UPH:Q", title="UPH", format=".2f"),
                ],
            )
            .properties(height=280)
        )
        st.altair_chart(chart, use_container_width=True)

        st.markdown("### 每日任务类型明细")

        task_detail_df = (
            task_trend_df[["date", "tasktype", "qty", "hours", "UPH"]]
            .sort_values(["date", "tasktype"])
            .copy()
        )
        task_detail_df = task_detail_df.rename(
            columns={
                "date": "日期",
                "tasktype": "任务类型",
                "qty": "件量",
                "hours": "工时(h)",
                "UPH": "UPH",
            }
        )
        st.dataframe(task_detail_df, use_container_width=True)

        task_csv = task_detail_df.to_csv(index=False).encode("utf-8-sig")
        task_file_name = f"tasktype_detail_{start_date}_{end_date}.csv"
        st.download_button(
            "导出明细（CSV）",
            data=task_csv,
            file_name=task_file_name,
            mime="text/csv",
            use_container_width=True,
        )


# ===== 人员页 =====
with tab_staff:
    # ===== Overall 排行榜：每天排名 + 区间平均 =====
    st.subheader("人员排行榜（Overall：按日排名 + 区间平均）")

    tmp = staff_daily_uph_f.copy()
    if tmp.empty:
        st.info("当前筛选条件下没有人员数据。")
    else:
        tmp["date_str"] = tmp["date"].astype(str)

        # 每天的排名透视成列
        pivot_rank = tmp.pivot_table(
            index=["staffid", "staff_name"],
            columns="date_str",
            values="rank",
            aggfunc="first",
        )

        # 区间平均排名 & 区间平均 Overall UPH
        avg_rank = tmp.groupby(["staffid", "staff_name"])["rank"].mean().round(1)
        avg_uph = tmp.groupby(["staffid", "staff_name"])["total_uph"].mean().round(2)

        result = pivot_rank.copy()
        result["区间平均排名"] = avg_rank
        result["区间平均Overall UPH"] = avg_uph

        # 排序
        result = result.sort_values("区间平均排名").reset_index()

        # 把日期列名改成「MM-DD 排名」
        rename_cols = {}
        for c in result.columns:
            if c in ["staffid", "staff_name", "区间平均排名", "区间平均Overall UPH"]:
                continue
            rename_cols[c] = c[5:] + " 排名"  # 例如 11-25 排名
        result = result.rename(columns=rename_cols)

        display_df = result.drop(columns=["staffid"]).rename(
            columns={"staff_name": "姓名"}
        )

        st.dataframe(display_df, use_container_width=True)
        st.caption(
            "说明：每一列日期为该员工在当天 Overall UPH 中的排名；"
            "右侧为在当前筛选日期区间内的平均排名和平均 Overall UPH。"
        )

        # 用区间平均 Overall UPH 做一个对比条形图
        st.markdown("### 区间平均 Overall UPH 对比")
        chart_df = display_df[["姓名", "区间平均Overall UPH"]].set_index("姓名")
        st.bar_chart(chart_df, height=260)

        # 在排行榜里选一个人，个人详情页自动带入
        selected_from_rank = st.selectbox(
            "查看员工详情",
            display_df["姓名"].tolist(),
        )
        st.session_state["selected_staff"] = selected_from_rank

    st.markdown("---")
    st.subheader("按任务类型的人员排行榜（按日排名 + 区间平均）")

    # ===== 按任务类型的排行榜：每天排名 + 区间平均 =====
    if per_person_task_f.empty:
        st.info("当前筛选条件下没有任务明细数据。")
    else:
        for ttype in WORK_TYPES:
            sub = per_person_task_f[per_person_task_f["TaskType"] == ttype].copy()
            if sub.empty:
                continue

            # 先按 天 × 人 汇总该任务的件量 & 工时
            daily = (
                sub.groupby(["Date", "StaffID", "StaffName"], as_index=False)
                .agg(
                    qty=("TotalQty", "sum"),
                    hours=("TaskHours", "sum"),
                )
            )
            daily = daily[daily["hours"] > 0]
            if daily.empty:
                continue

            daily["UPH"] = daily["qty"] / daily["hours"]
            daily["UPH"] = daily["UPH"].round(2)

            # 每天按 UPH 排名
            daily["rank"] = daily.groupby("Date")["UPH"].rank(
                ascending=False, method="min"
            )

            daily["date_str"] = daily["Date"].astype(str)
            pivot_rank = daily.pivot_table(
                index=["StaffID", "StaffName"],
                columns="date_str",
                values="rank",
                aggfunc="first",
            )

            avg_rank = daily.groupby(["StaffID", "StaffName"])["rank"].mean().round(1)
            avg_uph = daily.groupby(["StaffID", "StaffName"])["UPH"].mean().round(2)

            result = pivot_rank.copy()
            result["区间平均排名"] = avg_rank
            result["区间平均UPH"] = avg_uph

            result = result.sort_values("区间平均排名").reset_index()

            rename_cols = {}
            for c in result.columns:
                if c in ["StaffID", "StaffName", "区间平均排名", "区间平均UPH"]:
                    continue
                rename_cols[c] = c[5:] + " 排名"
            result = result.rename(columns=rename_cols)

            display_df = result.drop(columns=["StaffID"]).rename(
                columns={"StaffName": "姓名"}
            )

            st.markdown(f"#### {ttype}")
            st.dataframe(display_df, use_container_width=True)


# ===== 个人详情页 =====
with tab_detail:
    st.subheader("个人详情")

    all_staff = (
        staff_daily_uph["staff_name"]
        .dropna()
        .drop_duplicates()
        .sort_values()
        .tolist()
    )
    if not all_staff:
        st.info("当前没有人员数据。")
    else:
        # 如果在排行榜里选过人，就默认选中那个人
        default_name = st.session_state.get("selected_staff")
        if default_name in all_staff:
            default_index = all_staff.index(default_name)
        else:
            default_index = 0

        sel_staff = st.selectbox(
            "选择一位员工",
            all_staff,
            index=default_index,
        )
        st.session_state["selected_staff"] = sel_staff

        staff_daily_one = staff_daily_uph[
            (staff_daily_uph["staff_name"] == sel_staff)
            & (staff_daily_uph["date"] >= start_date)
            & (staff_daily_uph["date"] <= end_date)
        ].copy()

        if staff_daily_one.empty:
            st.info("该员工在所选日期内没有任务数据。")
        else:
            # 1）该员工在区间内的每日 Overall UPH 趋势
            st.markdown(f"### {sel_staff} 每日 Overall UPH（所选区间）")
            staff_daily_one = staff_daily_one.sort_values("date")
            st.line_chart(
                staff_daily_one.set_index("date")[["total_uph"]],
                height=260,
            )

            # 2）选择某一天，查看「各环节表现及排名」+ 饼图
            available_dates = sorted(staff_daily_one["date"].unique().tolist())
            sel_date = st.date_input(
                "选择要查看详情的日期（该员工有数据的日期）",
                value=available_dates[-1],
                min_value=available_dates[0],
                max_value=available_dates[-1],
            )

            st.markdown(f"### {sel_staff} 在 {sel_date} 各环节表现及排名")

            # 当天所有人的环节明细（使用按日期过滤后的 per_person_task_f）
            day_seg = (
                per_person_task_f[
                    per_person_task_f["Date"] == sel_date
                ]
                .dropna(subset=["Segment"])
                .groupby(["StaffID", "StaffName", "Segment"], as_index=False)
                .agg(
                    TotalQty=("TotalQty", "sum"),
                    Hours=("TaskHours", "sum"),
                )
            )
            day_seg = day_seg[day_seg["Hours"] > 0]
            if day_seg.empty:
                st.info("该日期没有环节明细数据。")
            else:
                day_seg["UPH"] = (day_seg["TotalQty"] / day_seg["Hours"]).round(2)

                # 对每个环节，算排名
                day_seg["RankInSegment"] = (
                    day_seg.groupby("Segment")["UPH"]
                    .rank(ascending=False, method="min")
                )
                # 每个环节多少人
                seg_counts = (
                    day_seg.groupby("Segment", as_index=False)["StaffID"]
                    .nunique()
                    .rename(columns={"StaffID": "SegmentStaffCount"})
                )
                day_seg = day_seg.merge(seg_counts, on="Segment", how="left")

                # 该员工当日各环节记录
                staff_seg_day = day_seg[day_seg["StaffName"] == sel_staff].copy()

                if staff_seg_day.empty:
                    st.info("该员工在所选日期没有环节数据。")
                else:
                    staff_seg_day_show = staff_seg_day[
                        ["Segment", "TotalQty", "Hours", "UPH", "RankInSegment", "SegmentStaffCount"]
                    ].rename(
                        columns={
                            "Segment": "环节",
                            "TotalQty": "件量",
                            "Hours": "工时(h)",
                            "UPH": "UPH",
                            "RankInSegment": "该环节排名",
                            "SegmentStaffCount": "参与人数",
                        }
                    )
                    staff_seg_day_show = staff_seg_day_show.sort_values("环节")
                    st.dataframe(staff_seg_day_show, use_container_width=True)

                    # 3）饼图：该员工当天各环节工时占比 / 件量占比
                    st.markdown("### 当日各环节工时占比 / 件量占比")

                    # 工时饼图
                    pie_hours_df = staff_seg_day[["Segment", "Hours"]].copy()
                    pie_hours_df = pie_hours_df.rename(
                        columns={"Segment": "环节", "Hours": "工时(h)"}
                    )

                    hours_chart = (
                        alt.Chart(pie_hours_df)
                        .mark_arc()
                        .encode(
                            theta=alt.Theta(field="工时(h)", type="quantitative"),
                            color=alt.Color(field="环节", type="nominal"),
                            tooltip=["环节", "工时(h)"],
                        )
                        .properties(title="各环节工时占比")
                    )

                    # 件量饼图
                    pie_qty_df = staff_seg_day[["Segment", "TotalQty"]].copy()
                    pie_qty_df = pie_qty_df.rename(
                        columns={"Segment": "环节", "TotalQty": "件量"}
                    )

                    qty_chart = (
                        alt.Chart(pie_qty_df)
                        .mark_arc()
                        .encode(
                            theta=alt.Theta(field="件量", type="quantitative"),
                            color=alt.Color(field="环节", type="nominal"),
                            tooltip=["环节", "件量"],
                        )
                        .properties(title="各环节件量占比")
                    )

                    col_pie1, col_pie2 = st.columns(2)
                    with col_pie1:
                        st.altair_chart(hours_chart, use_container_width=True)
                    with col_pie2:
                        st.altair_chart(qty_chart, use_container_width=True)

            # 4）该员工在所选区间内按环节拆分的 UPH 趋势
            st.markdown("### 该员工在所选区间内按环节拆分的 UPH 趋势")

            seg_one = (
                per_person_task_f[
                    per_person_task_f["StaffName"] == sel_staff
                ]
                .dropna(subset=["Segment"])
                .groupby(["Date", "Segment"], as_index=False)
                .agg(TotalQty=("TotalQty", "sum"), Hours=("TaskHours", "sum"))
            )
            if seg_one.empty:
                st.info("该员工在所选区间内没有环节数据。")
            else:
                seg_one["UPH"] = seg_one["TotalQty"] / seg_one["Hours"]
                seg_one = seg_one.sort_values(["Date", "Segment"])
                seg_one["Date_ts"] = pd.to_datetime(seg_one["Date"])
                seg_one["Date_str"] = seg_one["Date_ts"].dt.strftime("%m-%d")

                seg_chart = (
                    alt.Chart(seg_one)
                    .mark_line(point=True)
                    .encode(
                        x=alt.X(
                            "Date_str:N",
                            title="日期",
                            axis=alt.Axis(labelAngle=0),
                        ),
                        y=alt.Y("UPH:Q", title="UPH"),
                        color=alt.Color("Segment:N", title="环节"),
                        tooltip=[
                            alt.Tooltip("Date_ts:T", title="日期"),
                            alt.Tooltip("Segment:N", title="环节"),
                            alt.Tooltip("TotalQty:Q", title="件量", format=".0f"),
                            alt.Tooltip("Hours:Q", title="工时(h)", format=".2f"),
                            alt.Tooltip("UPH:Q", title="UPH", format=".2f"),
                        ],
                    )
                    .properties(height=260)
                )
                st.altair_chart(seg_chart, use_container_width=True)

            # 5）任务类型分布（区间总量）
            st.markdown("### 任务类型分布（区间总件量占比）")
            task_dist = (
                per_person_task_f[
                    per_person_task_f["StaffName"] == sel_staff
                ]
                .groupby("TaskType", as_index=False)
                .agg(TotalQty=("TotalQty", "sum"))
            )
            if task_dist.empty:
                st.info("无任务类型数据。")
            else:
                task_dist["占比"] = task_dist["TotalQty"] / task_dist["TotalQty"].sum()
                st.dataframe(task_dist, use_container_width=True)

                dist_chart = (
                    alt.Chart(
                        task_dist.rename(columns={"TaskType": "任务类型"})
                    )
                    .mark_arc()
                    .encode(
                        theta=alt.Theta(field="TotalQty", type="quantitative"),
                        color=alt.Color(field="任务类型", type="nominal"),
                        tooltip=["任务类型", "TotalQty", "占比"],
                    )
                    .properties(title="区间各任务类型件量占比")
                )
                st.altair_chart(dist_chart, use_container_width=True)
