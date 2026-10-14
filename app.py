from __future__ import annotations
import io, time, os
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import streamlit as st

try:
    import yfinance as yf
except Exception:
    yf = None

st.set_page_config(page_title="إعصار أحمد | ماسح الأسهم", layout="wide")
st.title("🌪️ إعصار أحمد — ماسح الأسهم")
st.caption("نسخة مجانية أولية. بيانات Yahoo Finance عبر yfinance؛ التغطية والتأخير وحدود الطلبات تختلف حسب الرمز والسوق. ليست خدمة بيانات لحظية مضمونة.")

TF = {"5 دقائق": ("5m", "60d"), "30 دقيقة": ("30m", "60d"), "ساعة": ("60m", "730d"), "يومي": ("1d", "max"), "أسبوعي": ("1wk", "max")}
EXP_LOOKBACK = 500
CONTRA_LOOKBACK = 200
DR_LEN, DR_STD, DR_KC1 = 20, 2.5, 0.5

def wma(s, n):
    w = np.arange(1, n+1, dtype=float)
    return s.rolling(n, min_periods=n).apply(lambda x: np.dot(x, w)/w.sum(), raw=True)

def rma(s, n):
    # Wilder RMA with SMA seed
    a = s.to_numpy(dtype=float)
    out = np.full(len(a), np.nan)
    if len(a) < n: return pd.Series(out, index=s.index)
    seed = np.nanmean(a[:n])
    if np.isnan(seed): return pd.Series(out, index=s.index)
    out[n-1] = seed
    prev = seed
    for i in range(n, len(a)):
        if np.isnan(a[i]): continue
        prev = (prev*(n-1)+a[i])/n
        out[i] = prev
    return pd.Series(out, index=s.index)

def tornado(df):
    if df is None or df.empty: return pd.DataFrame()
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [str(x).lower() for x in df.columns]
    required = ["open","high","low","close","volume"]
    if not all(x in df.columns for x in required): return pd.DataFrame()
    df = df[required].dropna(subset=["open","high","low","close"]).sort_index()
    if df.empty: return pd.DataFrame()
    o,h,l,c = [df[x].astype(float).to_numpy() for x in ["open","high","low","close"]]
    hac = (o+h+l+c)/4
    hao = np.full(len(df), np.nan)
    hao[0] = (o[0]+c[0])/2
    for i in range(1,len(df)): hao[i]=(hao[i-1]+hac[i-1])/2
    hah = np.maximum.reduce([h,hao,hac]); hal=np.minimum.reduce([l,hao,hac])
    ha_close = pd.Series(hac,index=df.index); ha_high=pd.Series(hah,index=df.index); ha_low=pd.Series(hal,index=df.index)
    basis=wma(ha_close,DR_LEN)
    dev=ha_close.rolling(DR_LEN,min_periods=DR_LEN).std(ddof=0)*DR_STD
    upbb=basis+dev; lobb=basis-dev
    bbw=((upbb-lobb)/basis.replace(0,np.nan)).fillna(0)
    prev=ha_close.shift(1)
    tr=pd.concat([ha_high-ha_low,(ha_high-prev).abs(),(ha_low-prev).abs()],axis=1).max(axis=1)
    atr=rma(tr,DR_LEN)
    upkc=basis+DR_KC1*atr; lokc=basis-DR_KC1*atr
    sqz=(upbb<upkc)&(lobb>lokc)
    exp=bbw >= bbw.rolling(EXP_LOOKBACK,min_periods=1).max()
    contra=bbw <= bbw.rolling(CONTRA_LOOKBACK,min_periods=1).min()
    sig=((exp.shift(1).fillna(False)&~exp) | (contra.shift(1).fillna(False)&~contra) | (sqz.shift(1).fillna(False)&~sqz))
    out=df.copy()
    out["ha_high"],out["ha_low"],out["ha_close"],out["bbw"],out["signal"]=ha_high,ha_low,ha_close,bbw,sig.fillna(False)
    return out

@st.cache_data(ttl=300, show_spinner=False)
def nasdaq_symbols():
    import requests
    urls = [
        "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt",
    ]
    rows=[]
    for url in urls:
        r=requests.get(url,timeout=20); r.raise_for_status()
        lines=r.text.splitlines()
        for line in lines:
            if "|" not in line or line.startswith("Symbol|") or line.startswith("File Creation Time"):
                continue
            p=line.split("|")
            if len(p)>3 and p[0] and p[0] not in ("Symbol",""):
                sym=p[0].strip()
                if sym.isascii() and not sym.startswith("$") and "^" not in sym and "/" not in sym:
                    rows.append(sym.replace(".","-"))
    return sorted(set(rows))

def normalize_yahoo(symbol, market):
    s=symbol.strip()
    if market=="السعودي":
        s=s.replace("TADAWUL:","").replace(".SR","")
        return s+".SR"
    s=s.replace("NASDAQ:","")
    return s

with st.sidebar:
    st.header("إعدادات الفحص")
    market_choice=st.multiselect("الأسواق",["السعودي","ناسداك"],default=["السعودي","ناسداك"])
    tfs=st.multiselect("الفواصل",list(TF.keys()),default=list(TF.keys()))
    mode=st.radio("قائمة الأسهم",["ناسداك تلقائي + قائمة سعودية مبدئية","رفع ملف CSV لقائمة الرموز"])
    max_symbols=st.number_input("حد أقصى للرموز في دورة الفحص",min_value=1,max_value=5000,value=100,step=25)
    st.warning("الفحص المجاني لآلاف الرموز على 5 فواصل قد يتجاوز حدود وقت التشغيل ومزود البيانات. ابدأ بعينة ثم زد العدد تدريجيًا.")

saudi_seed = ["2222.SR","1120.SR","2010.SR","1211.SR","7010.SR","1180.SR","1150.SR","2082.SR","4001.SR","4003.SR","2380.SR","2350.SR","2280.SR","1320.SR","1321.SR"]
if mode=="رفع ملف CSV لقائمة الرموز":
    uploaded=st.file_uploader("ملف CSV يحتوي عمود symbol وعمود market (SAUDI أو US)",type=["csv"])
    if uploaded:
        universe=pd.read_csv(uploaded)
    else:
        universe=pd.DataFrame(columns=["symbol","market"])
else:
    st.info("يتم جلب قائمة ناسداك من Nasdaq Trader عند الفحص، أما قائمة السعودية أدناه فهي عينة أولية وليست جميع أسهم تاسي ونمو.")
    universe=pd.DataFrame([{"symbol":s,"market":"SAUDI"} for s in saudi_seed])
    if "ناسداك" in market_choice:
        try:
            with st.spinner("تحميل قائمة رموز ناسداك..."):
                ns=nasdaq_symbols()
            universe=pd.concat([universe,pd.DataFrame([{"symbol":s,"market":"US"} for s in ns])],ignore_index=True)
            st.caption(f"رموز ناسداك المكتشفة: {len(ns):,}")
        except Exception as e:
            st.error(f"تعذر تحميل قائمة ناسداك: {e}")

if "السعودي" not in market_choice:
    universe=universe[universe["market"].astype(str).str.upper().ne("SAUDI")]
if "ناسداك" not in market_choice:
    universe=universe[universe["market"].astype(str).str.upper().ne("US")]

if not universe.empty:
    universe.columns=[str(c).lower().strip() for c in universe.columns]
    if not {"symbol","market"}.issubset(universe.columns):
        st.error("الملف يجب أن يحتوي عمودين: symbol وmarket.")
        universe=pd.DataFrame(columns=["symbol","market"])
    else:
        universe=universe.dropna(subset=["symbol"]).drop_duplicates("symbol")
        universe=universe.head(int(max_symbols))

st.metric("عدد الرموز في دورة الفحص", f"{len(universe):,}")
st.caption("الفحص لا يرسل أوامر تداول. تنبيه الإعصار يتحقق من معادلات المؤشر بعد إغلاق الشمعة المتاحة لدى المزود.")

if st.button("🚀 افحص الأسهم الآن",type="primary",disabled=(yf is None or universe.empty or not tfs)):
    results=[]; failures=[]; progress=st.progress(0)
    tasks=max(1,len(universe)*len(tfs)); done=0
    for _,row in universe.iterrows():
        market_raw=str(row["market"]).upper()
        market="السعودي" if market_raw in ("SAUDI","السعودي","TADAWUL") else "ناسداك"
        if market not in market_choice: continue
        symbol=str(row["symbol"]).strip()
        yahoo=normalize_yahoo(symbol,market)
        for tf_name in tfs:
            interval,period=TF[tf_name]
            done+=1
            progress.progress(min(1.0,done/tasks))
            try:
                bars=yf.download(yahoo,period=period,interval=interval,auto_adjust=False,progress=False,threads=False)
                if bars is None or bars.empty:
                    failures.append(f"{symbol} ({tf_name}): لا توجد بيانات متاحة من المصدر المجاني.")
                    continue
                calc=tornado(bars)
                if calc.empty or len(calc)<30:
                    failures.append(f"{symbol} ({tf_name}): تاريخ غير كافٍ.")
                    continue
                # Ignore current, potentially incomplete candle for intraday intervals.
                if tf_name in ("5 دقائق","30 دقيقة","ساعة") and len(calc)>1:
                    last=calc.iloc[-2]
                    stamp=calc.index[-2]
                else:
                    last=calc.iloc[-1]
                    stamp=calc.index[-1]
                if bool(last["signal"]):
                    results.append({
                        "السوق":market,"الرمز":symbol,"الفاصل":tf_name,
                        "وقت الشمعة":str(stamp),"السعر":float(last["close"]),
                        "أعلى HA":float(last["ha_high"]),"أدنى HA":float(last["ha_low"]),
                        "إشارة":"إعصار"
                    })
            except Exception as e:
                failures.append(f"{symbol} ({tf_name}): {e}")
            time.sleep(0.03)
    st.session_state["results"]=results
    st.session_state["failures"]=failures

results=st.session_state.get("results",[])
failures=st.session_state.get("failures",[])
st.subheader(f"النتائج المطابقة: {len(results)}")
if results:
    res=pd.DataFrame(results).sort_values(["السوق","الفاصل","الرمز"])
    st.dataframe(res,use_container_width=True,hide_index=True)
    st.download_button("تنزيل النتائج CSV",res.to_csv(index=False).encode("utf-8-sig"),"tornado_results.csv","text/csv")
else:
    st.info("اضغط «افحص الأسهم الآن» لبدء دورة الفحص. لن تظهر نتائج قبل جلب بيانات فعلية.")

if failures:
    with st.expander(f"رموز لم تُفحص أو تعذرت بياناتها ({len(failures)})"):
        st.write("\n".join(failures[:300]))

st.markdown("---")
st.markdown("**حدود النسخة المجانية:**")
st.markdown("""
- لا توجد ضمانة أن Yahoo Finance يوفر كل أسهم تاسي ونمو أو بياناتها على كل الفواصل.
- بيانات 5 و30 دقيقة محدودة تاريخيًا لدى كثير من المصادر المجانية، وقد تخضع لقيود طلبات.
- هذه النسخة لا تعمل كخدمة خلفية مستمرة؛ الفحص يبدأ عند الضغط على الزر. الاستضافة المجانية قد تنام عند عدم الاستخدام.
- نافذة اتساع 500 شمعة والانكماش 200 شمعة كما اتفقنا. يجب مطابقة النتائج مع TradingView على عينة قبل الاعتماد عليها.
""")
