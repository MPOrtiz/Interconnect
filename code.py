# Interconnect
Interconnect aims to forecast its customer churn rate. If a user is identified as planning to leave, they will be offered promotional codes and special plan options. Interconnect's marketing team has gathered some customer data, including information about their plans and contracts.
# TOC
* [Previewing information](#1)
* [Data Preprocessing](#2)
* [Data Exploring Analysis](#3)
* [Predictive Data Modeling](#4)
# Work Plan
1. Initialization
2. Data Preprocessing
3. Exploring Data Analysis
   - Graphs 
   - Levene Test
   - ANOVA Test
   - EDA Conclusions
4. Data Models
   - Dummy Model
   - HOE Models
       - Logistic Regression
       - Tree Classifier
       - Random Forest
   - Ordinal Encoding Model
       - Tree Classifier
       - Random Forest
       - Light GBM Classifier
       - Cat Boost Classifier
# 1
# 1. Initialization
import math

import lightgbm as lgb
import numpy as np
import pandas as pd
import seaborn as sns

from matplotlib import pyplot as plt

from scipy import stats as st

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import shuffle

from catboost import CatBoostClassifier
# 2
# 2. Data Preprocessing
2.1 Data Info
df_contract=pd.read_csv('/datasets/final_provider/contract.csv')
df_personal=pd.read_csv('/datasets/final_provider/personal.csv')
df_internet=pd.read_csv('/datasets/final_provider/internet.csv')
df_phone=pd.read_csv('/datasets/final_provider/phone.csv')
2.1.1 Contract Data Frame
print(df_contract.info())
print()
print(df_contract.head())
df_contract['TotalCharges'] = pd.to_numeric(df_contract['TotalCharges'], errors='coerce')
print(df_contract.info())
print('Null Data:')
print(df_contract.isna().sum())
print()
print('Duplicated Data:',df_contract.duplicated().sum())
df_contract['TotalCharges'].fillna(0,inplace=True)
print('Null Data:')
print(df_contract.isna().sum())
print()
print('Duplicated Data:',df_contract.duplicated().sum())
2.1.2 Personal Data Frame
print(df_personal.info())
print(df_personal.head())
print('Null Data:')
print(df_personal.isna().sum())
print()
print('Duplicated Data:',df_personal.duplicated().sum())
2.1.3 Internet Data Frame
print(df_internet.info())
print()
print(df_internet.head())
print('Null Data:')
print(df_internet.isna().sum())
print()
print('Duplicated Data:',df_internet.duplicated().sum())
2.1.4 Phone Data Frame
print(df_phone.info())
print()
print(df_phone.head())
print('Null Data:')
print(df_phone.isna().sum())
print()
print('Duplicated Data:',df_phone.duplicated().sum())
2.2 Columns Rename
print("Contract Columns: \n")
print(df_contract.columns)

new_cool_names =[]

for old_names in df_contract.columns:
    name_stripped=old_names.strip()
    name_lowered=name_stripped.lower()
    name_no_spaces=name_lowered.replace(" ","_")
    new_cool_names.append(name_no_spaces)

df_contract.columns=new_cool_names
print()
print(df_contract.columns)
print("Personal Columns: \n")
print(df_personal.columns)

new_cool_names =[]

for old_names in df_personal.columns:
    name_stripped=old_names.strip()
    name_lowered=name_stripped.lower()
    name_no_spaces=name_lowered.replace(" ","_")
    new_cool_names.append(name_no_spaces)

df_personal.columns=new_cool_names
print()
print(df_personal.columns)
print("Internet Columns: \n")
print(df_internet.columns)

new_cool_names =[]

for old_names in df_internet.columns:
    name_stripped=old_names.strip()
    name_lowered=name_stripped.lower()
    name_no_spaces=name_lowered.replace(" ","_")
    new_cool_names.append(name_no_spaces)

df_internet.columns=new_cool_names
print()
print(df_internet.columns)
print("Phone Columns: \n")
print(df_phone.columns)

new_cool_names =[]

for old_names in df_phone.columns:
    name_stripped=old_names.strip()
    name_lowered=name_stripped.lower()
    name_no_spaces=name_lowered.replace(" ","_")
    new_cool_names.append(name_no_spaces)

df_phone.columns=new_cool_names
print()
print(df_phone.columns)
2.3 Complete Data Set
df_infocontract=pd.merge(df_contract,df_personal,on=['customerid'],how='outer')
df_infocontract=pd.merge(df_infocontract,df_internet,on=['customerid'],how='outer')
df_infocontract=pd.merge(df_infocontract,df_phone,on=['customerid'],how='outer')

print(df_infocontract.info())
print()
print(df_infocontract.head(10))
df_infocontract['begindate'] = pd.to_datetime(df_infocontract['begindate'])
df_infocontract['begindate_year'] = df_infocontract['begindate'].dt.year
df_infocontract['begindate_month'] = df_infocontract['begindate'].dt.month
df_infocontract['begindate_day'] = df_infocontract['begindate'].dt.day
df_infocontract['enddate_datetime'] = pd.to_datetime(df_infocontract['enddate'], errors='coerce')
df_infocontract['churndate'] = df_infocontract['enddate_datetime'].dt.date.where(df_infocontract['enddate_datetime'].notna(), df_infocontract['enddate'])
df_infocontract['churndate_year'] = df_infocontract['enddate_datetime'].dt.year
df_infocontract['churndate_month'] = df_infocontract['enddate_datetime'].dt.month
df_infocontract['churndate_day'] = df_infocontract['enddate_datetime'].dt.day
print('Null Data:')
print(df_infocontract.isna().sum())
print()
print('Duplicated Data:',df_infocontract.duplicated().sum())
df_infocontract.fillna('N/A',inplace=True)
print('Null Data:')
print(df_infocontract.isna().sum())
print()
print('Duplicated Data:',df_infocontract.duplicated().sum())
print(df_infocontract)
The four datasets were reviewed to ensure the information was complete, with no duplicate or missing data. Column data types were reassigned to integers where necessary.  
Column names were converted to lowercase.  
The four datasets were merged into a single table to consolidate the information.
# 3
<div class="alert alert-block alert-success">
<b>Comentario del revisor (1ra Iteracion)</b> <a class=“tocSkip”></a>

Correcto, realizaste una exploración inical que deja ver claramente la calidad de los datos con los que se trabajará y el proceso aplicado para mejorarlos
</div>
# 3. Exploring Data Analysis
df_partner=df_infocontract.groupby('partner')['customerid'].count().reset_index()
print(df_partner)
print()

df_partenermontlycharge=df_infocontract.groupby('dependents')['monthlycharges'].mean().reset_index()
print(df_partenermontlycharge)
print()

df_dependents=df_infocontract.groupby('dependents')['customerid'].count().reset_index()
print(df_dependents)
print()
Graphs
df_perservices_internet = df_infocontract.groupby('internetservice')['customerid'].count().reset_index()
df_perservices_internet.rename(columns={'customerid': 'count'}, inplace=True)

df_perservices_phone = df_infocontract.groupby('multiplelines')['customerid'].count().reset_index()
df_perservices_phone.rename(columns={'customerid': 'count'}, inplace=True)

df_summary = pd.DataFrame({
    'service': ['Internet', 'N/A Internet', 'Telefono', 'N/A Telefono'],
    'count': [
        df_perservices_internet[df_perservices_internet['internetservice'].isin(['DSL', 'Fiber optic'])]['count'].sum(),
        df_perservices_internet[df_perservices_internet['internetservice'] == 'N/A']['count'].sum(),
        df_perservices_phone[df_perservices_phone['multiplelines'].isin(['Si', 'No'])]['count'].sum(),
        df_perservices_phone[df_perservices_phone['multiplelines'] == 'N/A']['count'].sum()
    ]
})

print(df_summary)
print()
plt.bar(df_summary['service'],df_summary['count'], align='center',alpha=0.65)
plt.ylabel('Percentage of services')
plt.xlabel('Service Type')
plt.title('Distribution of Services')
plt.title('Services Hired')
plt.show()
Most users subscribe to internet services more than phone services.
Gender
df_gender=df_infocontract.groupby('gender')['customerid'].count().reset_index()
print(df_gender)
print()
plt.bar(df_gender['gender'],df_gender['customerid'], align='center',alpha=0.65,color=['blue', 'orange'])
plt.title('Contracts by Gender')
plt.show()
The number of male and female users is very similar.
Senior Citizen
df_seniorcitizen=df_infocontract.groupby('seniorcitizen')['customerid'].count().reset_index()
print(df_seniorcitizen)
print()
plt.bar(df_seniorcitizen['seniorcitizen'],df_seniorcitizen['customerid'],  align='center',alpha=0.65,color=['blue', 'orange'])
plt.xticks([0, 1], ["Not Senior Citizen", "Senior Citizen"])
plt.xlabel("Senior Citizen (0 = No, 1 = Sí)")
plt.ylabel("Contracts Quantity")
plt.title("Contracts by Senior Citizen")
plt.show()
The majority of users are under 60 years old, suggesting that they are young individuals.
Contracts through the Years and Months
df_begin_year=df_infocontract.groupby('begindate_year',)['customerid'].count().reset_index()
print(df_begin_year)
print()

df_begin_yearmonth=df_infocontract.groupby(['begindate_year','begindate_month'])['customerid'].count().reset_index()
print(df_begin_yearmonth)
print()

df_begin_days=df_infocontract.groupby(['begindate_day'])['customerid'].count().reset_index()
print(df_begin_days)
print()

df_churn_yearmonth=df_infocontract.groupby(['churndate_year','churndate_month'])['customerid'].count().reset_index()
df_churn_yearmonth.replace('N/A', np.nan, inplace=True)
df_churn_yearmonth.dropna(axis=0, inplace=True)
print(df_churn_yearmonth)
print()

df_churn_days=df_infocontract.groupby(['churndate_day'])['customerid'].count().reset_index()
print(df_churn_days)
print()
print('Statistics Quantity of Customers')
print(df_begin_year['customerid'].describe())
plt.scatter(df_begin_year['begindate_year'], df_begin_year['customerid'], color='b', alpha=0.7, label="Contracts")
plt.title("Contracts through the years")
plt.plot(df_begin_year['begindate_year'], df_begin_year['customerid'], linestyle='-', color='b', alpha=0.5)
for year in df_begin_yearmonth['begindate_year'].unique():
    df_year = df_begin_yearmonth[df_begin_yearmonth['begindate_year'] == year]
    plt.plot(df_year['begindate_month'], df_year['customerid'], marker='o', linestyle='-', label=str(year))

plt.xlabel("Months")
plt.ylabel("Contract Quantity")
plt.title("Contracts through the years")
plt.xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], 
           ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'],
           rotation=45)
plt.legend(title="Año") 

# Mostrar gráfico
plt.show()

Yearly Contract Trends:
The number of contracts fluctuates over the years, with a significant peak in 2019, showing the highest contract volume.
There is a notable drop in 2020, which could be due to external factors such as economic downturns or global events.

Monthly Contract Trends:
The year 2014 saw a sharp increase in contracts in February, followed by a steep decline in the following months.
The year 2019 consistently had the highest number of contracts throughout most months compared to other years.
Other years, such as 2015–2018, show relatively stable trends with minor fluctuations.
The contracts in 2020 appear significantly lower, especially in the early months, which aligns with the observed drop in the yearly trend.

General Observations:
The data suggests that 2019 was a strong year for contracts, while 2020 experienced a sharp decline.
The early months of the year tend to have higher contract volumes in some years (e.g., 2014 and 2019).
The consistency of contract numbers from 2015 to 2018 indicates a relatively stable demand in those years.
Churn Date Through the years
for year in df_churn_yearmonth['churndate_year'].unique():
    df_year = df_churn_yearmonth[df_churn_yearmonth['churndate_year'] == year]
    plt.plot(df_year['churndate_month'], df_year['customerid'], marker='o', linestyle='-', label=str(year))

plt.xlabel("Months")
plt.ylabel("Contract Quantity")
plt.title("Churn Contracts through the years")
plt.xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], 
           ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'],
           rotation=45)
plt.legend(title="Año") 

plt.show()
Internet Services 
df_internetservices=df_infocontract.groupby(['internetservice'])['customerid'].count().reset_index()
df_internetservices.replace('N/A', np.nan, inplace=True)
df_internetservices.dropna(axis=0, inplace=True)
print(df_internetservices)
print()
plt.bar(df_internetservices['internetservice'],df_internetservices['customerid'], align='center',alpha=0.65,color=['blue', 'orange'])
plt.title('Internet Distribution')
plt.show()
Most people subscribe to fiber optic services.
Internet Services
columns_to_group = ['onlinesecurity', 'onlinebackup', 'deviceprotection', 'techsupport', 'streamingmovies']
column_names = ['OnlineSecurity', 'OnlineBack', 'DeviceProtection', 'TechSupport', 'Streaming']

df_grouped = {}

for col, name in zip(columns_to_group, column_names):
    df_grouped[name] = df_infocontract.groupby(col)['customerid'].count().reset_index().iloc[1:]

df_total = df_grouped['OnlineSecurity'].copy()  

for col, name in zip(columns_to_group[1:], column_names[1:]):  
    df_total = df_total.merge(df_grouped[name], left_on='onlinesecurity', right_on=col, how='left')

df_total = df_total.select_dtypes(include='number')

df_total.columns = column_names

print('Data Frame Internet Services')
print(df_total)
categories = df_total.columns 
values_yes = df_total.iloc[0].values 
values_no = df_total.iloc[1].values  

x = np.arange(len(categories))

plt.figure(figsize=(10, 6))
plt.bar(x - 0.2, values_yes, width=0.4, label="Hired", color='b', alpha=0.7)
plt.bar(x + 0.2, values_no, width=0.4, label="Not Hired", color='r', alpha=0.7)

plt.xticks(x, categories, rotation=45)
plt.ylabel("Quantity of users")
plt.title("Users who contracted additional internet services")
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.show()

General Trend:More users tend to subscribe to additional internet services rather than opting out, as indicated by the dominance of blue bars over red ones.
Most Popular Services:

Tech Support and Online Security appear to be the most frequently hired additional services, as they have the highest number of users subscribed.

Streaming services also have a high adoption rate, with nearly equal numbers of users hiring and not hiring.
Multiplelines Phone Services
df_multiplelines=df_infocontract.groupby('multiplelines')['customerid'].count().reset_index()
df_multiplelines.replace('N/A', np.nan, inplace=True)
df_multiplelines.dropna(axis=0, inplace=True)
print(df_multiplelines)
print()
plt.bar(df_multiplelines['multiplelines'],df_multiplelines['customerid'], align='center',alpha=0.65,color=['blue', 'orange'])
plt.title('Multiplelines Phone')
plt.show()
The number of people who subscribe to multiple lines and those who do not is very similar.
Contact Type
df_type=df_infocontract.groupby('type')['customerid'].count().reset_index()
print(df_type)
print()
plt.bar(df_type['type'],df_type['customerid'], align='center',alpha=0.65,color=['blue', 'orange','green'])
plt.title('Contracts by Type')
plt.show()
Preference for Month-to-Month Contracts
The majority of customers prefer month-to-month contracts. This suggests that users value flexibility and may avoid long-term commitments.

Potential Business Implications
The high number of month-to-month contracts may indicate a higher customer churn rate, as users can easily switch providers.
Contract's Paymentmethod
df_paymentmethod=df_infocontract.groupby('paymentmethod')['customerid'].count().reset_index()
print(df_paymentmethod)
print()

df_monthlycharges=df_infocontract.groupby('paymentmethod')['monthlycharges'].mean().reset_index()
print(df_monthlycharges)
print()

df_totalcharges=df_infocontract.groupby('paymentmethod')['totalcharges'].mean().reset_index()
print(df_totalcharges)
plt.figure(figsize=(9, 6))
plt.bar(df_paymentmethod['paymentmethod'],df_paymentmethod['customerid'], align='center',alpha=0.65,color=['blue', 'orange','green','purple'])
plt.title('Contracts by Paymentmethod')
plt.show()
Preference for Automatic Payments
The most commonly used payment methods are bank transfer (automatic) and credit card (automatic), with nearly the same number of contracts.
Most users prefer automated transactions for convenience and to avoid missed payments.
plt.figure(figsize=(10, 6))
plt.bar(df_totalcharges['paymentmethod'],df_totalcharges['totalcharges'], align='center',alpha=0.65,color=['blue', 'orange','green','purple'])
plt.title('Contracts by Paymentmethod')
plt.show()
Box Plots
df_payment_bp=df_infocontract.groupby(['paymentmethod','begindate_year'])['monthlycharges'].sum().reset_index()
print('Sales by Method Payment')
#print(df_payment_bp)
plt.figure(figsize=(10, 10))
plt.title('Sales by Method Payment')
sns.boxplot(data=df_payment_bp, x='paymentmethod', y='monthlycharges')

Q1 = df_payment_bp['monthlycharges'].quantile(0.25)  
Q3 = df_payment_bp['monthlycharges'].quantile(0.75)
IQR = Q3 - Q1

lower_limit = Q1 - (1.5 * IQR)
upper_limit = Q3 + (1.5 * IQR)

df_no_outliers = df_payment_bp[(df_payment_bp['monthlycharges'] >= lower_limit) & (df_payment_bp['monthlycharges'] <= upper_limit)]

# Visualizar el boxplot sin outliers
plt.figure(figsize=(10, 10))
plt.title('Sales by Method Payment')
sns.boxplot(data=df_no_outliers, x='paymentmethod', y='monthlycharges')  
plt.show()

df_payments_s=df_infocontract.groupby('paymentmethod')[['monthlycharges','totalcharges']].describe()
print('Montly Charges')
print(df_payments_s['monthlycharges'])
print()
print('Total Charges')
print(df_payments_s['totalcharges'])

correlation_periodpayment = df_monthlycharges['monthlycharges'].corr(df_totalcharges['totalcharges'])
print()
print('Correlation between periods of payments')
print(correlation_periodpayment)
print()
Higher Revenue from Automatic Payments:
The bank transfer (automatic) and credit card (automatic) payment methods generate the highest revenues.
This indicates that customers using automatic payments may have longer retention periods or higher spending habits.

Business Insights:
Encouraging customers to switch to automatic payment methods could help stabilize revenue and reduce payment delays.
Strong revenue from automatic payments suggests that subscription-based or recurring payment models are more profitable.
ANOVA Test
df_payment_bk=df_payment_bp[df_payment_bp['paymentmethod']=='Bank transfer (automatic)']
df_payment_cc=df_payment_bp[df_payment_bp['paymentmethod']=='Credit card (automatic)']
df_payment_ec=df_payment_bp[df_payment_bp['paymentmethod']=='Electronic check']
df_payment_mc=df_payment_bp[df_payment_bp['paymentmethod']=='Mailed check']

#Calculo de la varianza Bank Transfer
variance_bk=np.var(df_payment_bk['monthlycharges'])
print('Mean Montly Charges Bank Transfer',df_payment_bk['monthlycharges'].mean())
print('Variance Montly Charges Bank Transfer:',variance_bk)
print()

#Calculo de la varianza Credit Card
variance_cc=np.var(df_payment_cc['monthlycharges'])
print('Mean Montly Charges Credit card',df_payment_cc['monthlycharges'].mean())
print('Variance Montly Credit card:',variance_cc)
print()

#Calculo de la varianza Electronic Check
variance_ec=np.var(df_payment_ec['monthlycharges'])
print('Mean Montly Charges Electronic check',df_payment_ec['monthlycharges'].mean())
print('Variance Montly Electronic check:',variance_ec)
print()

#Calculo de la varianza Mailed Check
variance_mc=np.var(df_payment_mc['monthlycharges'])
print('Mean Montly Charges Mailed check',df_payment_mc['monthlycharges'].mean())
print('Variance Montly Mailed check',variance_mc)
print()
#Levene Test
# Hipótesis nula: Las varianzas de los grupos son iguales.
# Hipótesis alternativa: Al menos uno de los grupos tiene una variable diferente.

# Prueba de Levene para igualdad de varianzas

pvalue_levene_payment = st.levene(df_payment_bk['monthlycharges'],df_payment_cc['monthlycharges'],df_payment_ec['monthlycharges'],df_payment_mc['monthlycharges'])

# Imprimir resultados de la prueba de Levene
print('Statistic',pvalue_levene_payment.statistic)
print('P value', pvalue_levene_payment.pvalue)

if pvalue_levene_payment.pvalue < 0.05:
    print('Se rechaza la hipótesis nula: Al menos uno de los grupos tiene una variable diferente')
else:
    print('No podemos rechazar la hipótesis nula : Las varianzas de los grupos son iguales')
alpha = 0.05

result_anova = st.f_oneway(
    df_payment_bk['monthlycharges'],
    df_payment_cc['monthlycharges'],
    df_payment_ec['monthlycharges'],
    df_payment_mc['monthlycharges']
)

print("P value:", result_anova.pvalue)

if result_anova.pvalue < alpha:
    print("Rechazamos la hipótesis nula: Al menos un grupo es diferente.")
else:
    print("No podemos rechazar la hipótesis nula: No hay evidencia de diferencias significativas entre los grupos.")

Hyphotesis Test
#Levene Test
# Hipótesis nula: Las varianzas de los grupos son iguales.
# Hipótesis alternativa: Al menos uno de los grupos tiene una variable diferente.

# Prueba de Levene para igualdad de varianzas

pvalue_levene_payment = st.levene(df_payment_bk['monthlycharges'],df_payment_mc['monthlycharges'])

# Imprimir resultados de la prueba de Levene
print('Statistic',pvalue_levene_payment.statistic)
print('P value', pvalue_levene_payment.pvalue)

if pvalue_levene_payment.pvalue < 0.05:
    print('The null hypothesis is rejected: At least one of the groups has a different variable.')
else:
    print('We cannot reject the null hypothesis: The variances of the groups are equal.')
pvalue_levene_payment = st.levene(df_payment_bk['monthlycharges'], df_payment_mc['monthlycharges'])

equal_var = pvalue_levene_payment.pvalue >= 0.05

t_stat, p_value_ttest = st.ttest_ind(df_payment_bk['monthlycharges'], df_payment_mc['monthlycharges'], equal_var=equal_var)

print(f"Result t-Student:")
print(f"Stat t: {t_stat:.4f}")
print(f"P-value: {p_value_ttest:.4f}")

alpha = 0.05
if p_value_ttest < alpha:
    print("The null hypothesis is rejected: There is a significant difference in the group means.")
else:
    print("The null hypothesis cannot be rejected: There is not enough evidence to claim that the means are different.")

There is no statistical evidence to suggest that revenue differs significantly between the payment methods (Bank Transfer, Credit Card, Electronic Check, and Mailed Check). 

This implies that customers generate similar revenues regardless of their payment method.
<div class="alert alert-block alert-success">
<b>Comentario del revisor (1ra Iteracion)</b> <a class=“tocSkip”></a>

Muy bien realizado el EDA, revisaste las diferentes variables a utilizar con gráficas y conclusiones sobre las mismas lo que demuestra tu entendimiento de estas, además el aplicar prueba de hipótesis es una buena práctica para no realizar procesos a ciegas sino con argumentos claros en los siguientes pasos
</div>
# 4
# 4. Predictive Data Modeling
Data Frames Clean for Predictives Models
df_clean=df_infocontract.drop(['customerid','begindate','enddate','paperlessbilling','begindate_day','enddate_datetime','begindate_month','churndate_year','churndate_month','churndate_day'],axis=1)
df_clean['statuschurn'] = np.where(df_clean['churndate'] == "No", "0", "1")
df_clean=df_clean.drop('churndate',axis=1)
(df_clean.info())
print('Duplicated Data:',df_clean.duplicated().sum())
df_clean.drop_duplicates(inplace=True)
print('Duplicated Data:',df_clean.duplicated().sum())
print(df_clean.describe())
categorical_features=['type','paymentmethod','gender','seniorcitizen','partner','dependents','internetservice','onlinesecurity','onlinebackup','deviceprotection','techsupport','streamingtv','streamingmovies','multiplelines']

for features in categorical_features:
    print(features)
    print(df_clean[features].value_counts())
    print()
print(df_clean.info())
data_types=pd.DataFrame(df_clean.dtypes)
print(data_types)
variable_categorical=list(data_types[data_types[0]=='object'].index)
print(variable_categorical)
variable_numerical=list(data_types[data_types[0]!='object'].index)
print(variable_numerical)
# Dummy Model
df_clean_ohe_dummy=df_clean
df_clean_ohe_dummy=pd.get_dummies(df_clean_ohe_dummy,drop_first=True)

features_dummy = df_clean_ohe_dummy.drop(['statuschurn_1'],axis=1)
target_dummy = df_clean_ohe_dummy['statuschurn_1']
features_train_dummy, features_test_dummy, target_train_dummy, target_test_dummy = train_test_split(features_dummy, target_dummy, test_size=0.2, random_state=12345)
features_train_dummy, features_valid_dummy, target_train_dummy, target_valid_dummy = train_test_split(features_train_dummy, target_train_dummy, test_size=0.25, random_state=42) 

dummy_model = DummyClassifier(strategy="constant", constant=1)
dummy_model.fit(features_train_dummy, target_train_dummy)

pred_train_dummy = dummy_model.predict(features_train_dummy)
pred_valid_dummy = dummy_model.predict(features_valid_dummy)
pred_test_dummy = dummy_model.predict(features_test_dummy)

accuracy_valid_dummy = accuracy_score(target_valid_dummy, pred_valid_dummy)
f1_valid_dummy = f1_score(target_valid_dummy, pred_valid_dummy)
auc_valid_dummy = roc_auc_score(target_valid_dummy, pred_valid_dummy)

print("\n Dummy Model (Target = 1)")
print(f" Accuracy (Validation): {accuracy_valid_dummy:.4f}")
print(f" F1-Score (Validation): {f1_valid_dummy:.4f}")
print(f" AUC-ROC (Validation): {auc_valid_dummy:.4f}")

# OHE MODELS
1. Decision Tree Classifier
df_clean_ohe=df_clean
df_clean_ohe=pd.get_dummies(df_clean_ohe,drop_first=True)
print(df_clean_ohe)
index_train_valid,index_test=train_test_split(df_clean.index,test_size=0.2,random_state=12345)
index_train,index_valid=train_test_split(index_train_valid,test_size=0.25,random_state=54321)
df_train=df_clean.loc[index_train]
df_valid=df_clean.loc[index_valid]
df_test=df_clean.loc[index_test]
df_ohe_train=df_clean_ohe.loc[index_train]
df_ohe_valid=df_clean_ohe.loc[index_valid]
df_ohe_test=df_clean_ohe.loc[index_test]
print('Training',df_train.shape)
print('Valid',df_valid.shape)
print('Testing',df_test.shape)
print('Training',df_ohe_train.shape)
print('Valid',df_ohe_valid.shape)
print('Testing',df_ohe_test.shape)
<div class="alert alert-block alert-success">
<b>Comentario del revisor (1ra Iteracion)</b> <a class=“tocSkip”></a>

Perfecto, tienes el proceso listo para pasar al modelado
</div>
Split Features and Target
#Trainning
features_train_ohe=df_ohe_train.drop(['statuschurn_1'],axis=1)
target_train_ohe=df_ohe_train['statuschurn_1']
#Valid
features_valid_ohe=df_ohe_valid.drop(['statuschurn_1'],axis=1)
target_valid_ohe=df_ohe_valid['statuschurn_1']
#Testing
features_test_ohe=df_ohe_test.drop(['statuschurn_1'],axis=1)
target_test_ohe=df_ohe_test['statuschurn_1']
scaler=StandardScaler()
features_train_ohe[variable_numerical]=scaler.fit_transform(features_train_ohe[variable_numerical])
features_valid_ohe[variable_numerical]=scaler.transform(features_valid_ohe[variable_numerical])
features_test_ohe[variable_numerical]=scaler.transform(features_test_ohe[variable_numerical])
Class Imbalance
print(target_train_ohe.value_counts(normalize=True))
target_train_ohe.value_counts(normalize=True).plot(kind='bar')
<div class="alert alert-block alert-success">
<b>Comentario del revisor (1ra Iteracion)</b> <a class=“tocSkip”></a>

Muy bien, siempre en problemas de clasificación hay que revisar el balance de las clases para saber que técnicas aplicar durante la experimentación
</div>
def upsample(features, target, repeat):
    features_zeros = features[target == 0]
    features_ones = features[target == 1]
    target_zeros = target[target == 0]
    target_ones = target[target == 1]

    features_upsampled = pd.concat([features_zeros] + [features_ones] * repeat)
    target_upsampled = pd.concat([target_zeros] + [target_ones] * repeat)
    features_upsampled, target_upsampled = shuffle(features_upsampled, target_upsampled, random_state=12345)
    
    return features_upsampled, target_upsampled

features_upsampled, target_upsampled = upsample(features_train_ohe, target_train_ohe, 3)
for depth in range(1, 11):
    model_tree = DecisionTreeClassifier(random_state=12345, max_depth=depth)
    model_tree.fit(features_upsampled, target_upsampled)
    score_train = model_tree.score(features_train_ohe, target_train_ohe)
    predicted_valid = model_tree.predict(features_valid_ohe)
    
    score_valid = model_tree.score(features_valid_ohe, target_valid_ohe)
    
    print(f"\nMaximum tree depth: {depth}")
    print('Accuracy Training Set: ',model_tree.score(features_train_ohe, target_train_ohe))
    print('Accuracy Validation Set: ', model_tree.score(features_valid_ohe, target_valid_ohe))
    print('Accuracy Testing Set: ', model_tree.score(features_test_ohe, target_test_ohe))
    print('F1:', f1_score(target_valid_ohe, predicted_valid))
    print('ROC-AUC:',roc_auc_score(target_valid_ohe,predicted_valid))  
2. Logistic Regression
model_lr = LogisticRegression()
model_lr.fit(features_train_ohe, target_train_ohe)
pred_train_lr = model_lr.predict(features_train_ohe)
pred_valid_lr=model_lr.predict(features_valid_ohe)
pred_test_lr=model_lr.predict(features_test_ohe)

score_train = model_lr.score(features_train_ohe,target_train_ohe) 
score_valid = model_lr.score(features_valid_ohe,target_valid_ohe)
score_test = model_lr.score(features_test_ohe,target_test_ohe)

print("Accuracy del modelo de regresión logística en el conjunto de entrenamiento:", score_train)
print("Accuracy del modelo de regresión logística en el conjunto de validación:", score_valid)
print("Accuracy del modelo de regresión logística en el conjunto de entrenamiento:", score_test)
print()

print('Valid')
print('F1:', f1_score(target_valid_ohe, pred_valid_lr))
print('ROC-AUC',roc_auc_score(target_valid_ohe,predicted_valid))
print()
print('Test')
print('F1:', f1_score(target_test_ohe,pred_test_lr))
print('ROC-AUC',roc_auc_score(target_test_ohe,pred_test_lr))
3. Random Forest
print('Random Forest Balanced Class:\n')

def upsample(features, target, repeat):
    features_zeros = features[target == 0]
    features_ones = features[target == 1]
    target_zeros = target[target == 0]
    target_ones = target[target == 1]

    features_upsampled = pd.concat([features_zeros] + [features_ones] * repeat)
    target_upsampled = pd.concat([target_zeros] + [target_ones] * repeat)
    features_upsampled, target_upsampled = shuffle(features_upsampled, target_upsampled, random_state=12345)
    
    return features_upsampled, target_upsampled

features_upsampled, target_upsampled = upsample(features_train_ohe, target_train_ohe, 3)

#Modelo de bosque
print('Random Forest:\n')

for n_tree in range (10,110,10):
    model_forest=RandomForestClassifier(n_estimators=n_tree,random_state=12345)
    model_forest.fit(features_upsampled, target_upsampled)
    predictions_valid=model_forest.predict(features_valid_ohe)

    print('Trees Numbers:',n_tree)
    print('Training',model_forest.score(features_train_ohe, target_train_ohe))
    print('Valid', model_forest.score(features_train_ohe,target_train_ohe))
    print('Testing', model_forest.score(features_test_ohe,target_test_ohe))
    print('F1:', f1_score(target_valid_ohe,predictions_valid))
    print('ROC-AUC',roc_auc_score(target_valid_ohe,predictions_valid))
    print()
for n_tree in range (1,20,2):
    model_forest=RandomForestClassifier(n_estimators=n_tree,random_state=12345)
    model_forest.fit(features_upsampled, target_upsampled)
    predictions_valid=model_forest.predict(features_valid_ohe)

    print('Tree Number:',n_tree)
    print('Training',model_forest.score(features_train_ohe, target_train_ohe))
    print('Valid', model_forest.score(features_train_ohe,target_train_ohe))
    print('Testing', model_forest.score(features_test_ohe,target_test_ohe))
    print('F1:', f1_score(target_valid_ohe,predictions_valid))
    print('ROC-AUC',roc_auc_score(target_valid_ohe,predictions_valid))
    print()
# Ordinal Encoding Model
print(df_clean.info())

df_clean[categorical_features]=OrdinalEncoder().fit_transform(df_clean[categorical_features])
print(df_clean)
1. Tree Classifier
df_clean['statuschurn'] = df_clean['statuschurn'].astype('int64')
features = df_clean.drop(columns=['statuschurn'])  
target = df_clean['statuschurn']

features_train, features_test, target_train, target_test = train_test_split(features, target, test_size=0.2, random_state=12345)
features_train, features_valid, target_train, target_valid = train_test_split(features_train, target_train, test_size=0.2, random_state=12345)

print('Training')
print(features_train.shape)
print(target_train.shape)

print('Validation')
print(features_valid.shape)
print(target_valid.shape)

print('Test')
print(features_test.shape)
print(target_test.shape)
print("Distribución del target en training:")
print(target_train.value_counts(normalize=True))
target_train.value_counts(normalize=True).plot(kind='bar')
def upsample(features, target, repeat):
    features_zeros = features[target == 0]
    features_ones = features[target == 1]
    target_zeros = target[target == 0]
    target_ones = target[target == 1]

    features_upsampled = pd.concat([features_zeros] + [features_ones] * repeat)
    target_upsampled = pd.concat([target_zeros] + [target_ones] * repeat)
    features_upsampled, target_upsampled = shuffle(features_upsampled, target_upsampled, random_state=12345)
    
    return features_upsampled, target_upsampled
if target_train.value_counts(normalize=True).max() > 0.7:
    print("Upsampling Applied")
    features_upsampled, target_upsampled = upsample(features_train, target_train, 3)
for depth in range(1, 11):
    model_tree = DecisionTreeClassifier(random_state=12345, max_depth=depth)
    model_tree.fit(features_train, target_train)
    
    predicted_train = model_tree.predict(features_train)
    predicted_valid = model_tree.predict(features_valid)
    predicted_test = model_tree.predict(features_test)
    
    score_train = accuracy_score(target_train, predicted_train)
    score_valid = accuracy_score(target_valid, predicted_valid)
    score_test = accuracy_score(target_test, predicted_test)
    f1_valid = f1_score(target_valid, predicted_valid)
    roc_auc_valid = roc_auc_score(target_valid, model_tree.predict_proba(features_valid)[:, 1]) 
    
    print(f"\n Maximum tree depth:: {depth}")
    print(f" Accuracy (Train): {score_train:.4f}")
    print(f" Accuracy (Validation): {score_valid:.4f}")
    print(f" Accuracy (Test): {score_test:.4f}")
    print(f" F1-Score: {f1_valid:.4f}")
    print(f" ROC-AUC: {roc_auc_valid:.4f}")
2. Random Forest
print('Random Forest:\n')

for n_tree in range (10,110,10):
    model_forest=RandomForestClassifier(n_estimators=n_tree,random_state=12345)
    model_forest.fit(features_upsampled, target_upsampled)
    predictions_valid=model_forest.predict(features_valid)

    print('Tree Number:',n_tree)
    print('Training',model_forest.score(features_train, target_train))
    print('Valid', model_forest.score(features_valid,target_valid))
    print('Testing', model_forest.score(features_test,target_test))
    print('F1:', f1_score(target_valid,predictions_valid))
    print('ROC-AUC',roc_auc_score(target_valid,predictions_valid))
    print()
print('Random Forest:\n')

for n_tree in range (1,15,2):
    model_forest=RandomForestClassifier(n_estimators=n_tree,random_state=12345)
    model_forest.fit(features_upsampled, target_upsampled)
    predictions_valid=model_forest.predict(features_valid)

    print('Tree Number:',n_tree)
    print('Training',model_forest.score(features_train, target_train))
    print('Valid', model_forest.score(features_valid,target_valid))
    print('Testing', model_forest.score(features_test,target_test))
    print('F1:', f1_score(target_valid,predictions_valid))
    print('ROC-AUC',roc_auc_score(target_valid,predictions_valid))
    print()
3. Light GBM Classifier
model_lgb = lgb.LGBMClassifier(n_estimators=1000, verbose=1, metric='auc')
model_lgb.fit(features_train, target_train, eval_set=(features_valid, target_valid), early_stopping_rounds=50)
pred_train_lgb = model_lgb.predict_proba(features_train)[:, 1]
pred_valid_lgb = model_lgb.predict_proba(features_valid)[:, 1]
pred_test_lgb = model_lgb.predict_proba(features_test)[:, 1]

pred_train_binary = (pred_train_lgb >= 0.5).astype(int)
pred_valid_binary = (pred_valid_lgb >= 0.5).astype(int)
pred_test_binary = (pred_test_lgb >= 0.5).astype(int)
auc_train = roc_auc_score(target_train, pred_train_lgb)
auc_valid = roc_auc_score(target_valid, pred_valid_lgb)
auc_test = roc_auc_score(target_test, pred_test_lgb)

accuracy_train = accuracy_score(target_train, pred_train_binary)
accuracy_valid = accuracy_score(target_valid, pred_valid_binary)
accuracy_test = accuracy_score(target_test, pred_test_binary)

f1_valid = f1_score(target_valid, pred_valid_binary)
logloss_valid = log_loss(target_valid, pred_valid_lgb)
print(f"\n AUC-ROC Train: {auc_train:.4f}")
print(f" AUC-ROC Valid: {auc_valid:.4f}")
print(f" AUC-ROC Test: {auc_test:.4f}")
print()
print(f" Accuracy Train: {accuracy_train:.4f}")
print(f" Accuracy Valid: {accuracy_valid:.4f}")
print(f" Accuracy Test: {accuracy_test:.4f}")
print()
print(f" F1-Score Valid: {f1_valid:.4f}")
print(f" Log Loss Valid: {logloss_valid:.4f}")
3.1.2 Modelo Light GBM Classifier with categorical features
model_lgb2=lgb.LGBMClassifier(num_iterations=1000,verbose=1,metric='auc')
model_lgb2.fit(features_train,target_train,eval_set=(features_train,target_train),early_stopping_rounds=50,categorical_feature=categorical_features)
pred_train_lgb2 = model_lgb2.predict_proba(features_train)[:, 1]
pred_valid_lgb2 = model_lgb2.predict_proba(features_valid)[:, 1]
pred_test_lgb2 = model_lgb2.predict_proba(features_test)[:, 1]

pred_train_binary2 = (pred_train_lgb2 >= 0.5).astype(int)
pred_valid_binary2 = (pred_valid_lgb2 >= 0.5).astype(int)
pred_test_binary2 = (pred_test_lgb2 >= 0.5).astype(int)
auc_train2 = roc_auc_score(target_train, pred_train_lgb)
auc_valid2 = roc_auc_score(target_valid, pred_valid_lgb)
auc_test2 = roc_auc_score(target_test, pred_test_lgb)

accuracy_train2 = accuracy_score(target_train, pred_train_binary2)
accuracy_valid2 = accuracy_score(target_valid, pred_valid_binary2)
accuracy_test2 = accuracy_score(target_test, pred_test_binary2)

f1_valid2 = f1_score(target_valid, pred_valid_binary2)
logloss_valid2 = log_loss(target_valid, pred_valid_lgb2)
print(f"\n AUC-ROC Train: {auc_train2:.4f}")
print(f" AUC-ROC Valid: {auc_valid2:.4f}")
print(f" AUC-ROC Test: {auc_test2:.4f}")
print()
print(f" Accuracy Train: {accuracy_train2:.4f}")
print(f" Accuracy Valid: {accuracy_valid2:.4f}")
print(f" Accuracy Test: {accuracy_test2:.4f}")
print()
print(f" F1-Score Valid: {f1_valid2:.4f}")
print(f" Log Loss Valid: {logloss_valid2:.4f}")
4. Cat Boost
model_cat = CatBoostClassifier(iterations=2000, learning_rate=0.1,loss_function='Logloss', eval_metric='AUC',verbose=100)
model_cat.fit(features_train, target_train,eval_set=(features_valid, target_valid),use_best_model=True,early_stopping_rounds=50)
pred_train_cat = model_cat.predict_proba(features_train)[:, 1]
pred_valid_cat = model_cat.predict_proba(features_valid)[:, 1]
pred_test_cat = model_cat.predict_proba(features_test)[:, 1]

pred_train_binary_cat = (pred_train_cat >= 0.5).astype(int)
pred_valid_binary_cat = (pred_valid_cat >= 0.5).astype(int)
pred_test_binary_cat = (pred_test_cat >= 0.5).astype(int)
auc_train_cat = roc_auc_score(target_train, pred_train_cat)
auc_valid_cat = roc_auc_score(target_valid, pred_valid_cat)
auc_test_cat = roc_auc_score(target_test, pred_test_cat)

accuracy_train_cat = accuracy_score(target_train, pred_train_binary)
accuracy_valid_cat = accuracy_score(target_valid, pred_valid_binary)
accuracy_test_cat = accuracy_score(target_test, pred_test_binary)

f1_valid_cat = f1_score(target_valid, pred_valid_binary_cat)
logloss_valid_cat = log_loss(target_valid, pred_valid_cat)

print(f"\n AUC-ROC Train: {auc_train_cat:.4f}")
print(f" AUC-ROC Valid: {auc_valid_cat:.4f}")
print(f" AUC-ROC Test: {auc_test_cat:.4f}")
print()
print(f" Accuracy Train: {accuracy_train_cat:.4f}")
print(f" Accuracy Valid: {accuracy_valid_cat:.4f}")
print(f" Accuracy Test: {accuracy_test_cat:.4f}")
print()
print(f" F1-Score Valid: {f1_valid_cat:.4f}")
print(f" Log Loss Valid: {logloss_valid_cat:.4f}")

Dummy Model (Target = 1): 
Dummy Model is a weak benchmark and does not provide valuable predictive power.

One Hot Encoding
Logistic Regression: performs well but does not outperform the Decision Tree Classifier in F1-score or ROC-AUC.
Decision Tree Classifier: with depth 7 is the best model overall, providing high generalization performance.
Random Forest: models tend to overfit, achieving nearly perfect training accuracy, which is not ideal for real-world applications.

Ordinal Encoding
Tree Classifier is the simplest but weakest in terms of predictive power.
Random Forest improves performance but suffers from overfitting.
LightGBM is superior to both in terms of ROC-AUC and generalization, making it the best choice if interpretability is not a major concern.

The best model is CatBoost, as it outperforms all other models in ROC-AUC, F1-Score, and Log Loss. It provides the most balanced and robust performance across all datasets.

-MP Ortiz
