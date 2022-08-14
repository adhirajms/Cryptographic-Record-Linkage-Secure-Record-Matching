import streamlit as st
import pandas as pd
from faker import Faker
import seaborn as sns
import recordlinkage
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import matplotlib.pyplot as plt
from recordlinkage.datasets import load_febrl4
from recordlinkage.preprocessing import phonetic
from recordlinkage import Compare
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score
import plotly.figure_factory as ff
import plotly.graph_objects as go
import plotly.express as px
header=st.container()
dataset=st.container()
features=st.container()
model_training=st.container()
faker_data=st.container()
from snorkel.labeling import labeling_function
from snorkel.labeling import PandasLFApplier
from snorkel.labeling import LFAnalysis
from snorkel.labeling.model import MajorityLabelVoter
from snorkel.labeling.model import LabelModel

st.set_option('deprecation.showPyplotGlobalUse', False)
st.markdown(
        f"""
<style>
    .reportview-container .main .block-container{{
        max-width: 90%;
        padding-top: 5rem;
        padding-right: 5rem;
        padding-left: 5rem;
        padding-bottom: 5rem;
    }}
    img{{
    	max-width:40%;
    	margin-bottom:40px;
    }}
</style>
""",
        unsafe_allow_html=True,
    )
NOMATCH = 0
UNKNOWN = -1
MATCH = 1
@labeling_function()
def soc_sec(x):
    return MATCH if x.soc_sec_id>0.9 else UNKNOWN
@labeling_function()    
def soc_sec_rev(x):
    return NOMATCH if x.soc_sec_id<0.6 else UNKNOWN


@labeling_function()
def name(x):
    return MATCH if  x.given_name>0.9 and x.surname>0.9 else UNKNOWN

@labeling_function()
def name_date(x):
    return MATCH if  x.given_name>0.9 and x.surname>0.9 and x.date_of_birth>0.7 else UNKNOWN

@labeling_function()
def name_rev(x):
    return NOMATCH if  x.given_name<0.6 or x.surname<0.6 else UNKNOWN    

@labeling_function()
def address(x):
    return MATCH if  x.address>0.6 else UNKNOWN

@labeling_function()
def address_rev(x):
    return NOMATCH if  x.address<0.2 else UNKNOWN   

def plot_metrics(model,metrics_list,x_test,y_test):
    if "Confusion Matrix" in metrics_list:
        st.subheader("Confusion Matrix")
        plot_confusion_matrix(model, x_test, y_test, display_labels=class_names)
        st.pyplot()
    if "ROC Curve" in metrics_list:
        st.subheader("ROC Curve")
        plot_roc_curve(model, x_test, y_test)
        st.pyplot()
    if "Precision-Recall Curve" in metrics_list:
        st.subheader("Precision-Recall Curve")
        plot_precision_recall_curve(model, x_test, y_test)
        st.pyplot()
def data1(dfA,dfB,blocker=""):
      indexer = recordlinkage.Index()
      if blocker!="":
        indexer.block(blocker)
        candidate_links = indexer.index(dfA, dfB)
      else:
        a=list(dfA.index)
        b=list(dfB.index)
        candidate_links=pd.MultiIndex.from_product([a,b])  
      compare = Compare()
      compare.exact('phonetic_given_name', 'phonetic_given_name', label="phonetic_given_name")
      compare.exact('phonetic_surname', 'phonetic_surname', label='phonetic_surname')
      compare.string('given_name', 'given_name', method='jarowinkler', label="given_name")
      compare.string('surname', 'surname', method='jarowinkler', label="surname")
      compare.string('suburb', 'suburb', method='jarowinkler', label="suburb")
      compare.string('state', 'state', method='jaro_winkler', label="state")
      compare.string('address', 'address', method='cosine', label="address")
      compare.string('address_1', 'address_1', method='jarowinkler', label="address_1")
      compare.string('address_2', 'address_2', method='jarowinkler', label="address_2")
      compare.string("soc_sec_id","soc_sec_id",method='jarowinkler', label="soc_sec_id")
      compare.string("date_of_birth","date_of_birth",method='jarowinkler', label="date_of_birth")
      features = compare.compute(candidate_links, dfA, dfB)
      return features
  
fake = Faker(42)   
def data_creation(entries):
    given_name = []
    surname = []
    street_number=[]
    address_1=[]
    address_2=[]
    suburb=[]
    state = []
    postcode = []
    date_of_birth = []
    soc_sec_id = []
    
    for q in range(entries):
        given_name.append(fake.first_name())
        surname.append(fake.last_name())
        street_number.append(fake.building_number())
        address_1.append(fake.street_suffix())
        address_2.append(fake.street_name())
        suburb.append(fake.city())
        state.append(fake.state())
        postcode.append(fake.zipcode())
        soc_sec_id.append(fake.ssn())
        date_of_birth.append(fake.date_of_birth())
        
    df = pd.DataFrame(list(zip(given_name, surname, street_number, address_1, address_2, suburb,  postcode,state,date_of_birth,soc_sec_id)), 
                      columns= ['given_name', 'surname', 'street_number', 'address_1', 'address_2', 'suburb','postcode', 'state','date_of_birth','soc_sec_id'])
    return df  
class_names = ["No Match", "Match"]

st.sidebar.title("Interactive") 


models=st.sidebar.selectbox("How would you like to data to be modeled?",("Gradient Boosting", "Logistic Regression", "Weak Supervision"))

metrics = st.sidebar.multiselect("What metrics to plot?", ("Confusion Matrix", "ROC Curve", "Precision-Recall Curve"))

@st.cache()
def data():
    dfA, dfB, true_links = load_febrl4(return_links=True)
    dfA["phonetic_given_name"] = phonetic(dfA["given_name"], "soundex")
    dfB["phonetic_given_name"] = phonetic(dfB["given_name"], "soundex")
    dfA["phonetic_surname"] = phonetic(dfA["surname"], "soundex")
    dfB["phonetic_surname"] = phonetic(dfB["surname"], "soundex")
    dfA["initials"] = (dfA["given_name"].str[0]  + dfA["surname"].str[0])
    dfB["initials"] = (dfB["given_name"].str[0]  + dfB["surname"].str[0])
    dfA["date_of_birth"] = dfA["date_of_birth"].str.replace('-', "")
    dfB["date_of_birth"] = dfB["date_of_birth"].str.replace('-', "")
    dfA["soc_sec_id"] = dfA["soc_sec_id"].str.replace('-', "")
    dfB["soc_sec_id"] = dfB["soc_sec_id"].str.replace('-', "")
    dfA['address']=dfA['street_number']+" "+dfA['address_1']+" "+dfA['address_2']
    dfB['address']=dfB['street_number']+" "+dfB['address_1']+" "+dfB['address_2'] 
    return dfA,dfB,true_links 
with header:
	st.title("Welcome to Record Linkage")
    
    

with dataset:
    st.header("Data from Record Linkage Package")
    dfA, dfB, true_links=data()
    st.markdown("Few Lines of Data")
    st.write(dfA.head(5))  

with  features:
    st.header("Modelling Features")
    features=data1(dfA,dfB,"initials")
    features['Target']=features.index.isin(true_links)
    features['Target']=features['Target'].astype(int)
    
    data=features.reset_index(drop=True)
    X=data.drop(['Target'],axis=1)
    Y=data['Target']
    X_train,X_test,y_train,y_test=train_test_split(X,Y,random_state=42,test_size=0.2)
    st.markdown("Input train Dataset")
    st.write(X_train[:5])
    st.markdown("Actual Output train Dataset")
    st.write(y_train[:5])
    y_train_dst=y_train.value_counts()
    st.markdown("Matches and Non Matches Distribution in Train Data")
    st.write(y_train_dst)
    
with  model_training:
    if models=="Gradient Boosting":
        st.header("Applying Gradient Boosting to Model")
        n_estimators=st.slider("What would be the number of estimators of the model?", min_value=10,max_value=100,value=10,step=10)
        max_depth=st.slider("What would be the max_depth of the model?", min_value=1,max_value=10,value=1,step=1)
        clf = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=0.1,max_depth=max_depth, random_state=42).fit(X_train, y_train)
        y_pred=clf.predict(X_test)
        plot1=pd.DataFrame()
        plot1['Features']=list(X_train.columns)
        plot1['Importance']=clf.feature_importances_
        plot1 = plot1.set_index('Features')
        st.bar_chart(plot1)
        model_final=clf
        st.markdown("Performance of Model on Test Data")
        y_pred=model_final.predict(X_test)
        accuracy = model_final.score(X_test, y_test)
        y_pred = model_final.predict(X_test)
        st.write("Accuracy: ", accuracy.round(2))
        st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
        st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2)) 
        plot_metrics(model_final,metrics,X_test,y_test)        
    elif models=="Logistic Regression":
        st.header("Applying Logistic Regression to Model")
        penalty=st.select_slider('Select penalty type',options=['l1', 'l2', 'elasticnet', 'none'],value=('l2'))
        C=st.slider("What would be the value of C(Inverse of regularization strength)?", min_value=0.1,max_value=2.0,value=1.0,step=0.1)
        lr=LogisticRegression(class_weight="balanced",penalty=penalty,C=C,solver="saga", l1_ratio=0.5)
        lr.fit(X_train,y_train)
        y_pred=lr.predict(X_train)
        y_pred1=lr.predict(X_test)
        plot1=pd.DataFrame()
        plot1['Features']=list(X_train.columns)
        plot1['Importance']=lr.coef_[0]
        plot1 = plot1.set_index('Features')
        st.bar_chart(plot1)
        model_final=lr
        st.markdown("Performance of Model on Test Data")
        y_pred=model_final.predict(X_test)
        accuracy = model_final.score(X_test, y_test)
        y_pred = model_final.predict(X_test)
        st.write("Accuracy: ", accuracy.round(2))
        st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
        st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2)) 
        plot_metrics(model_final,metrics,X_test,y_test)
    elif models=="Weak Supervision":
       st.header("Applying Snorkel to Model")
       data=features
       df = data.sample(frac=1)
       X_train=df.head(int(len(df)*0.8))
       X_test=df.tail(int(len(df)*0.2)) 
       lfs = [soc_sec, address,soc_sec_rev, name_rev, address_rev,name_date]
       applier = PandasLFApplier(lfs=lfs)
       L_train = applier.apply(df=X_train)
       st.write(LFAnalysis(L=L_train, lfs=lfs).lf_summary())
       L_test = applier.apply(df=X_test)
       model_type=st.select_slider('Select model type',options=['MajorityLabelVoter', 'LabelModel'],value=('LabelModel'))
       if model_type=="MajorityLabelVoter":
           model_final = MajorityLabelVoter()
           accuracy = model_final.score(L=L_test, Y=X_test.Target, tie_break_policy="random")["accuracy"]
           st.write("Accuracy: ", accuracy.round(2))
       else:
           model_final = LabelModel(cardinality=3, verbose=True)
           model_final.fit(L_train=L_train, n_epochs=500, log_freq=100, seed=42)
           accuracy = model_final.score(L=L_test, Y=X_test.Target, tie_break_policy="abstain")["accuracy"]
           st.write("Accuracy: ", accuracy.round(2))
       sol=model_final.predict(L_test)
       X_test['CLASS']=sol
       X_test=X_test.loc[(X_test.CLASS==0) | (X_test.CLASS==1)]
       if "Confusion Matrix" in metrics:
           st.subheader("Confusion Matrix")
           cm=confusion_matrix(X_test.Target,X_test.CLASS)
           cm=cm/len(X_test.CLASS)
           fig = plt.figure(figsize=(10, 8))
           sns.heatmap(cm*100, annot=True,annot_kws={"size": 16})
           st.pyplot(fig)
           

with faker_data:
    st.header("Running Model on Faker Data")
    sample_size=st.slider("What would be the sample size of Fake Data?", min_value=1000,max_value=15000,value=5000,step=1000)
    data_sample=data_creation(entries=sample_size)
    dfA1=data_sample
    dfB1=data_sample
    dfA1["phonetic_given_name"] = phonetic(dfA1["given_name"], "soundex")
    dfB1["phonetic_given_name"] = phonetic(dfB1["given_name"], "soundex")
    dfA1["phonetic_surname"] = phonetic(dfA1["surname"], "soundex")
    dfB1["phonetic_surname"] = phonetic(dfB1["surname"], "soundex")
    dfA1["initials"] = (dfA1["given_name"].str[0]  + dfA1["surname"].str[0])
    dfB1["initials"] = (dfB1["given_name"].str[0]  + dfB1["surname"].str[0])
    dfA1["date_of_birth"] = dfA1["date_of_birth"].astype(str).str.replace('-', "")
    dfB1["date_of_birth"] = dfB1["date_of_birth"].astype(str).str.replace('-', "")
    dfA1["soc_sec_id"] = dfA1["soc_sec_id"].astype(str).str.replace('-', "")
    dfB1["soc_sec_id"] = dfB1["soc_sec_id"].astype(str).str.replace('-', "")
    dfA1['address']=dfA1['street_number']+" "+dfA1['address_1']+" "+dfA1['address_2']
    dfB1['address']=dfB1['street_number']+" "+dfB1['address_1']+" "+dfB1['address_2']
    features1=data1(dfA1,dfB1,"initials")
    
    Match_Rate=st.slider("What would be the Probabilty Match Rate?", min_value=0.1,max_value=1.0,value=0.1,step=0.1)
    Display_Matches=st.slider("How many top Macthes to display?", min_value=1,max_value=10,value=5,step=1)
    if models=="Gradient Boosting" or models=="Logistic Regression":
        input1=features1
    else:
        L_fake = applier.apply(df=features1)
        input1=L_fake        
    features1['Match']=model_final.predict_proba(input1)[:,1]
    features1.reset_index(inplace=True)
    features1=features1[features1["level_0"]!=features1["level_1"]]
    show=features1[features1['Match']>=Match_Rate]
    st.header("Number of Matches")
    st.write(len(show)//2)
    show.sort_values(['Match'],ascending=False,inplace=True)
    show=show.reset_index(drop=True)
    display=0
    for i in range(0,len(show),2):
          display+=1
          if display==Display_Matches:
              break
          st.header("Probability of Matching")
          f1=show.iloc[i]
          st.write(f1["Match"]*100)
          d1=dfA1.iloc[[show['level_0'].values[i],show['level_1'].values[i]]]
          st.write(d1)