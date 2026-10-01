import argparse, json
import joblib, numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
from common import ROOT, dump, serve

FEATURES=['lag1','lag7','lag14','rolling7','rolling28','weekday_sin','weekday_cos','month_sin','month_cos','trend']

def generate():
    rng=np.random.default_rng(42);n=900;t=np.arange(n)
    dates=pd.date_range('2023-01-01',periods=n)
    sales=np.maximum(0,np.round(35+.018*t+9*np.sin(2*np.pi*t/7)+6*np.sin(2*np.pi*t/365)+rng.normal(0,4,n)))
    (ROOT/'data').mkdir(exist_ok=True)
    pd.DataFrame({'date':dates,'units':sales}).to_csv(ROOT/'data/daily_sales.csv',index=False)

def features(frame):
    dates=frame.date; y=frame.units
    out=pd.DataFrame(index=frame.index)
    for lag in [1,7,14]: out[f'lag{lag}']=y.shift(lag)
    for window in [7,28]: out[f'rolling{window}']=y.shift(1).rolling(window).mean()
    out['weekday_sin']=np.sin(2*np.pi*dates.dt.dayofweek/7)
    out['weekday_cos']=np.cos(2*np.pi*dates.dt.dayofweek/7)
    out['month_sin']=np.sin(2*np.pi*dates.dt.month/12)
    out['month_cos']=np.cos(2*np.pi*dates.dt.month/12)
    out['trend']=(dates-dates.iloc[0]).dt.days
    return out[FEATURES]

def read_data():
    frame=pd.read_csv(ROOT/'data/daily_sales.csv',parse_dates=['date']).sort_values('date').reset_index(drop=True)
    if len(frame)<180 or frame.date.duplicated().any() or frame.units.isna().any() or (frame.units<0).any():
        raise ValueError('Need >=180 unique daily rows and nonnegative units.')
    if not np.isfinite(frame.units).all() or not (frame.date.diff().dropna()==pd.Timedelta(days=1)).all():
        raise ValueError('Data must be finite, complete and equally spaced daily observations.')
    return frame

def metrics(y,p):
    return {'mae':float(mean_absolute_error(y,p)),'rmse':float(mean_squared_error(y,p)**.5),
        'wape':float(np.abs(np.array(y)-p).sum()/max(float(np.sum(y)),1))}

def recursive(model,history,days):
    frame=history.copy();pred=[]
    for _ in range(days):
        day=frame.date.iloc[-1]+pd.Timedelta(days=1)
        trial=pd.concat([frame,pd.DataFrame({'date':[day],'units':[np.nan]})],ignore_index=True)
        x=features(trial).iloc[[-1]]
        value=max(0,float(model.predict(x)[0]));pred.append(value)
        frame=pd.concat([frame,pd.DataFrame({'date':[day],'units':[value]})],ignore_index=True)
    return np.array(pred)

def train():
    if not (ROOT/'data/daily_sales.csv').exists():generate()
    frame=read_data();all_x=features(frame);valid=all_x.notna().all(axis=1)
    X=all_x[valid].reset_index(drop=True);y=frame.loc[valid,'units'].reset_index(drop=True)
    dev_X,test_X=X.iloc[:-60],X.iloc[-60:];dev_y,test_y=y.iloc[:-60],y.iloc[-60:]
    models={'ridge':make_pipeline(StandardScaler(),Ridge(alpha=10)),
        'gradient_boosting':HistGradientBoostingRegressor(max_iter=150,max_leaf_nodes=15,l2_regularization=5,random_state=42)}
    scores={};cv=TimeSeriesSplit(n_splits=4,test_size=45)
    for name,model in models.items():
        fold=[]
        for fit_idx,val_idx in cv.split(dev_X):
            model.fit(dev_X.iloc[fit_idx],dev_y.iloc[fit_idx]);fold.append(float(mean_absolute_error(dev_y.iloc[val_idx],np.maximum(0,model.predict(dev_X.iloc[val_idx])))))
        scores[name]={'fold_mae':fold,'mean_mae':float(np.mean(fold))}
    winner=min(scores,key=lambda k:scores[k]['mean_mae']);model=models[winner]
    model.fit(dev_X,dev_y)
    pred=np.maximum(0,model.predict(test_X))
    recursive_p=recursive(model,frame.iloc[:-60],60)
    report={'data':'synthetic single-product daily sales unless replaced','rows':len(frame),
        'validation':'Expanding time windows; one-step predictions use actual preceding-day observations.',
        'candidates':scores,'selected_model':winner,'test_last_60_days_one_step':metrics(test_y,pred),
        'seasonal_naive_lag7_one_step':metrics(test_y,test_X.lag7.to_numpy()),
        'test_last_60_days_recursive':metrics(test_y,recursive_p),
        'limitations':['Synthetic demand is not representative of a real shop.','Intervals are empirical rough bands, not calibrated coverage guarantees.',
            'One SKU only; no promotion, holidays, stockout censoring or price effects.','Recursive error can accumulate; evaluate the business forecast horizon.']}
    # Persist all-history model for future predictions, after the held-out evaluation.
    band=float(np.quantile(np.abs(test_y.to_numpy()-pred),.9))
    model.fit(X,y)
    (ROOT/'artifacts').mkdir(exist_ok=True)
    joblib.dump({'model':model,'rough_error_band':band},ROOT/'artifacts/model.joblib')
    dump(ROOT/'reports/metrics.json',report)
    pd.DataFrame({'date':frame.date.iloc[-60:].astype(str),'actual':test_y.to_numpy(),'one_step_prediction':pred,'recursive_prediction':recursive_p}).to_csv(ROOT/'reports/backtest.csv',index=False)
    return report

def dispatch(path,payload):
    if path=='/report':return json.loads((ROOT/'reports/metrics.json').read_text())
    if path!='/forecast':raise ValueError('Unknown endpoint')
    days=payload.get('days',14);lead=payload.get('lead_time_days',7);stock=payload.get('stock_on_hand',0)
    if type(days)!=int or not 1<=days<=60:raise ValueError('days must be an integer 1–60')
    if type(lead)!=int or not 1<=lead<=days:raise ValueError('lead_time_days must be 1–days')
    if isinstance(stock,bool) or not isinstance(stock,(int,float)) or not np.isfinite(stock) or stock<0:raise ValueError('stock_on_hand must be finite and nonnegative')
    bundle=joblib.load(ROOT/'artifacts/model.joblib');frame=read_data()
    pred=recursive(bundle['model'],frame,days);band=bundle['rough_error_band']
    rows=[{'date':str((frame.date.iloc[-1]+pd.Timedelta(days=i+1)).date()),'units':round(float(v),2),
        'rough_lower':round(max(0,float(v)-band),2),'rough_upper':round(float(v)+band,2)} for i,v in enumerate(pred)]
    demand=float(pred[:lead].sum());safety=band*np.sqrt(lead)
    return {'forecast':rows,'planning':{'expected_lead_time_demand':round(demand,2),'illustrative_safety_stock':round(safety,2),
        'recommended_order_units':max(0,int(np.ceil(demand+safety-stock)))},
        'interval_note':'Illustrative residual band; future coverage and independence are not guaranteed. Dates follow the data, not today.'}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['train','demo','serve']);args=p.parse_args()
    if args.command=='train':print(json.dumps(train(),indent=2))
    elif args.command=='demo':
        result=dispatch('/forecast',{'days':14,'stock_on_hand':150,'lead_time_days':7});dump(ROOT/'reports/demo.json',result);print(json.dumps(result,indent=2))
    else:serve(dispatch,8102)
