#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 24 15:44:46 2023

@author: admin
"""

import numpy as np
import pandas as pd

import config2

from config2 import SQLQuery

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from snowflake.sqlalchemy import URL


q = SQLQuery('snowflake')

# core libraries
import pickle
import pandas as pd
import numpy as np
import statsmodels.api as sm
import plotly.express as px
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings
warnings.filterwarnings("ignore")

# load modules
from preprocess import Convert,MissingValues,Outlier,FeatureSelection
from feature_transformation import Scaler,Transform,Selection
from model_building import split_test_train, feature_encoding, classification_models
from model_evaluations import model_metrics, feature_importance, probability_bins, cross_validation
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, Normalizer

# object initiation 
tf = Transform()
sel = Selection()
ft = FeatureSelection()
cv = Convert()
mv = MissingValues()
ot = Outlier()

# set seed
seed = 9


corr_arr = 0.5 # person correlation coefficient # change it to 0.5
vif_arr = 5 # vif coefficient
features_arr = 12 # total number of features to be selected from backward feature selection
iv_upper_limit = 0.6 # upper threshold of iv # change it to 0.6
iv_lower_limit = 0.02 # lower threshold of iv




plaid_txn_l1m = q(r"""select a.transaction_id, a.date, a.amount, a.plaid_account_id, a.description, a.merchant_name, a.type, a.categories, a.business_id, b.decision_date
                      from prod_db.adhoc.lending_dda_txns_w_owned_flag_final_20230625 a
                      
                      inner join (select *
                                  from prod_db.adhoc.lending_response_base_w_plaid_match_g20_txns_20230624) b 
                                  
                      on a.business_id = b.business_id
                      
                      where type = 'debit' 
                      
                      and date(a.date) < date(b.decision_date)
                      
                      and date(a.date) >= dateadd(day, -30, date(b.decision_date))
                      
                      """)
                                    
        
plaid_txn_l3m = q(r"""select a.transaction_id, a.date, a.amount, a.plaid_account_id, a.description, a.merchant_name, a.type, a.categories, a.business_id, b.decision_date
                      from prod_db.adhoc.lending_dda_txns_w_owned_flag_final_20230625 a
                      
                      inner join (select *
                                  from prod_db.adhoc.lending_response_base_w_plaid_match_g20_txns_20230624) b 
                                  
                      on a.business_id = b.business_id
                      
                      where type = 'debit' 
                      
                      and date(a.date) < date(b.decision_date)
                      
                      and date(a.date) >= dateadd(day, -90, date(b.decision_date))
                      
                      """)     
        
plaid_txn_l6m = q(r"""select a.transaction_id, a.date, a.amount, a.plaid_account_id, a.description, a.merchant_name, a.type, a.categories, a.business_id, b.decision_date
                      from prod_db.adhoc.lending_dda_txns_w_owned_flag_final_20230625 a
                      
                      inner join (select *
                                  from prod_db.adhoc.lending_response_base_w_plaid_match_g20_txns_20230624) b 
                                  
                      on a.business_id = b.business_id
                      
                      where type = 'debit' 
                      
                      and date(a.date) < date(b.decision_date)
                      
                      and date(a.date) >= dateadd(day, -180, date(b.decision_date))
                      
                      """)  

df1 = q("""select * from prod_db.adhoc.lending_response_base_w_plaid_match_g20_txns_20230624""")

plaid_txn_l1m['amount'] = plaid_txn_l1m['amount'].astype('float')

plaid_txn_l3m['amount'] = plaid_txn_l3m['amount'].astype('float')

plaid_txn_l6m['amount'] = plaid_txn_l6m['amount'].astype('float')

        
## Variables - last 3 months

plaid_txn_debit  = plaid_txn_l3m

# 1. Payroll Spend

df_pr = plaid_txn_debit[(plaid_txn_debit['categories'] == 'Transfer,Payroll') | (plaid_txn_debit['merchant_name'].isin(['ADP', 'Gusto', 'Paychex Tps', 'paychex', 'Paychex Eib']))]


grp1 = df_pr.groupby('business_id').agg(total_payroll_spend_l3m = ('amount', np.sum), avg_txn_amount_payroll_spend_l3m = ('amount',np.mean), txn_count_payroll_spend_l3m = ('transaction_id', 'count')).reset_index()

df = pd.merge(df1, grp1, how='left', on='business_id')


## 2. Rent

df_rent = plaid_txn_debit[plaid_txn_debit['categories'] == 'Payment,Rent']

grp2 = df_rent.groupby('business_id').agg(total_rent_spend_l3m = ('amount', np.sum), avg_txn_amount_rent_spend_l3m = ('amount',np.mean), txn_count_rent_spend_l3m = ('transaction_id', 'count')).reset_index()

df = pd.merge(df, grp2, how='left', on='business_id')



# 3. Utility bills

df_utility = plaid_txn_debit[(plaid_txn_debit['categories'].isin(['Service,Telecommunication Services', 
                                                                             'Service,Subscription',
                                                                             'Service,Utilities,Electric', 
                                                                             'Service,Utilities', 'Transfer,Billpay', 
                                                                             'Service,Utilities,Water', 
                                                                             'Service,Internet Services'])) 
                                   | (plaid_txn_debit['merchant_name'].isin(['Dominion Energy',
                                    'So Cal Gas','Xcel Energy','FPL','Eversource','National Grid',
                                    'Dukeenergy','PG&E','Spire','Atmos Energy','Southwest Gas',
                                    'FirstEnergy','NV Energy','So Cal Edison','DTE Energy']))]

grp3 = df_utility.groupby('business_id').agg(total_utility_spend_l3m = ('amount', np.sum), avg_txn_amount_utility_spend_l3m = ('amount',np.mean), count_txn_utility_spend_l3m = ('transaction_id', 'count')).reset_index()

df = pd.merge(df, grp3, how='left', on='business_id')


# 4. Marketing Spend

df_ad_spend = plaid_txn_debit[plaid_txn_debit['categories'].isin(['Service,Advertising and Marketing,Online Advertising', 'Service,Advertising and Marketing'])]

grp4 = df_ad_spend.groupby('business_id').agg(total_ad_spend_l3m = ('amount', np.sum), avg_txn_amount_ads_spend_l3m = ('amount',np.mean), count_txn_ads_spend_l3m = ('transaction_id', 'count')).reset_index()

df = pd.merge(df, grp4, how='left', on='business_id')



# 5. Credit card payment

df_cc_payment = plaid_txn_debit[plaid_txn_debit['categories'].isin(['Payment,Credit Card'])]

grp5 = df_cc_payment.groupby('business_id').agg(total_cc_payment_l3m = ('amount', np.sum), avg_txn_amount_cc_payment_l3m = ('amount',np.mean), count_txn_cc_payment_l3m = ('transaction_id', 'count')).reset_index()

df = pd.merge(df, grp5, how='left', on='business_id')



# 6. Personal spend

df_personal_spend = plaid_txn_debit[plaid_txn_debit['categories'].isin(['Shops,Supermarkets and Groceries', 'Shops,Sporting Goods',
                    'Travel,Gas Stations', 'Food and Drink,Restaurants', 'Shops,Digital Purchase', 'Shops,Computers and Electronics',
                    'Shops,Discount Stores', 'Travel,Parking', 'Service,Home Improvement,Swimming Pool Maintenance and Services', 'Shops,Bookstores',
                    'Shops,Warehouses and Wholesale Stores', 'Food and Drink,Restaurants,Fast Food',
                    'Shops,Jewelry and Watches', 'Food and Drink', 'Food and Drink,Restaurants,Coffee Shop', 'Shops,Food and Beverage Store', 'Service,Food and Beverage', 'Shops'])]


grp6 = df_personal_spend.groupby('business_id').agg(total_personal_spend_l3m = ('amount', np.sum), avg_txn_amount_personal_spend_l3m = ('amount',np.mean), count_txn_personal_spend_l3m = ('transaction_id', 'count')).reset_index()

df = pd.merge(df, grp6, how='left', on='business_id')



# 7. Service finance

df_service_finance = plaid_txn_debit[(plaid_txn_debit['categories'].isin(['Service', 'Service,Financial,Banking and Finance', 'Service,Financial', 'Service,Financial,Stock Brokers', 'Service,Financial,Loans and Mortgages',
                                                                                     'Service,Financial,Financial Planning and Investments', 'Service,Financial,Accounting and Bookkeeping',
                                                                                     'Service,Financial,Taxes', 'Service,Insurance'])) | 
                                           (plaid_txn_debit['merchant_name'].isin(['Internal Revenue Service',
                                                'IRS','Irs Usataxpymt','Franchise Tax Board', 'Irs'
                                                'Withheld','Intuit','Franchise Tax Bo','Federal Tax',
                                                'GEICO','State Farm','Progressive Insurance','Liberty Mutual',
                                                'Allstate','Primerica','Nationwide','Lemonade Insurance',
                                                'Transamerica','Northwestern Mutual']))]

grp7 = df_service_finance.groupby('business_id').agg(total_service_finance_l3m = ('amount', np.sum), avg_txn_amount_service_finance_l3m = ('amount',np.mean), count_txn_service_finance_l3m = ('transaction_id', 'count')).reset_index()

df = pd.merge(df, grp7, how='left', on='business_id')


# # 8. Gambling

# df_gambling = plaid_txn_debit[plaid_txn_debit['merchant_name'].isin(['BetMGM','Betfair','Pulsz',
#                                     'Golden Nugget','LuckyLand Slots','Bet365','UniBet','Performance Predictions',
#                                     'Bingo Cash','Everi','Clearwater Casino'])]

# grp8 = df_gambling.groupby('business_id').agg(total_gambling_merc_l3m = ('amount', np.sum), avg_txn_amount_gambling_merc_l3m = ('amount',np.mean), count_txn_gambling_merc_l3m = ('transaction_id', 'count')).reset_index()

# df = pd.merge(df, grp8, how='left', on='business_id')



## Repeat the same for 6 months

plaid_txn_debit  = plaid_txn_l6m

# 1. Payroll Spend

df_pr = plaid_txn_debit[(plaid_txn_debit['categories'] == 'Transfer,Payroll') | (plaid_txn_debit['merchant_name'].isin(['ADP', 'Gusto', 'Paychex Tps', 'paychex', 'Paychex Eib']))]

grp1 = df_pr.groupby('business_id').agg(total_payroll_spend_l6m = ('amount', np.sum), avg_txn_amount_payroll_spend_l6m = ('amount',np.mean), txn_count_payroll_spend_l6m = ('transaction_id', 'count')).reset_index()

df = pd.merge(df, grp1, how='left', on='business_id')


## 2. Rent

df_rent = plaid_txn_debit[plaid_txn_debit['categories'] == 'Payment,Rent']

grp2 = df_rent.groupby('business_id').agg(total_rent_spend_l6m = ('amount', np.sum), avg_txn_amount_rent_spend_l6m = ('amount',np.mean), txn_count_rent_spend_l6m = ('transaction_id', 'count')).reset_index()

df = pd.merge(df, grp2, how='left', on='business_id')



# 3. Utility bills

df_utility = plaid_txn_debit[(plaid_txn_debit['categories'].isin(['Service,Telecommunication Services', 
                                                                             'Service,Subscription',
                                                                             'Service,Utilities,Electric', 
                                                                             'Service,Utilities', 'Transfer,Billpay', 
                                                                             'Service,Utilities,Water', 
                                                                             'Service,Internet Services'])) 
                                   | (plaid_txn_debit['merchant_name'].isin(['Dominion Energy',
                                    'So Cal Gas','Xcel Energy','FPL','Eversource','National Grid',
                                    'Dukeenergy','PG&E','Spire','Atmos Energy','Southwest Gas',
                                    'FirstEnergy','NV Energy','So Cal Edison','DTE Energy']))]

grp3 = df_utility.groupby('business_id').agg(total_utility_spend_l6m = ('amount', np.sum), avg_txn_amount_utility_spend_l6m = ('amount',np.mean), count_txn_utility_spend_l6m = ('transaction_id', 'count')).reset_index()

df = pd.merge(df, grp3, how='left', on='business_id')


# 4. Marketing Spend

df_ad_spend = plaid_txn_debit[plaid_txn_debit['categories'].isin(['Service,Advertising and Marketing,Online Advertising', 'Service,Advertising and Marketing'])]

grp4 = df_ad_spend.groupby('business_id').agg(total_ad_spend_l6m = ('amount', np.sum), avg_txn_amount_ads_spend_l6m = ('amount',np.mean), count_txn_ads_spend_l6m = ('transaction_id', 'count')).reset_index()

df = pd.merge(df, grp4, how='left', on='business_id')



# 5. Credit card payment

df_cc_payment = plaid_txn_debit[plaid_txn_debit['categories'].isin(['Payment,Credit Card'])]

grp5 = df_cc_payment.groupby('business_id').agg(total_cc_payment_l6m = ('amount', np.sum), avg_txn_amount_cc_payment_l6m = ('amount',np.mean), count_txn_cc_payment_l6m = ('transaction_id', 'count')).reset_index()

df = pd.merge(df, grp5, how='left', on='business_id')



# 6. Personal spend

df_personal_spend = plaid_txn_debit[plaid_txn_debit['categories'].isin(['Shops,Supermarkets and Groceries', 'Shops,Sporting Goods',
                    'Travel,Gas Stations', 'Food and Drink,Restaurants', 'Shops,Digital Purchase', 'Shops,Computers and Electronics',
                    'Shops,Discount Stores', 'Travel,Parking', 'Service,Home Improvement,Swimming Pool Maintenance and Services', 'Shops,Bookstores',
                    'Shops,Warehouses and Wholesale Stores', 'Food and Drink,Restaurants,Fast Food',
                    'Shops,Jewelry and Watches', 'Food and Drink', 'Food and Drink,Restaurants,Coffee Shop', 
                    'Shops,Food and Beverage Store', 'Service,Food and Beverage', 'Shops'])]


grp6 = df_personal_spend.groupby('business_id').agg(total_personal_spend_l6m = ('amount', np.sum), avg_txn_amount_personal_spend_l6m = ('amount',np.mean), count_txn_personal_spend_l6m = ('transaction_id', 'count')).reset_index()

df = pd.merge(df, grp6, how='left', on='business_id')


# 7. Service finance

df_service_finance = plaid_txn_debit[(plaid_txn_debit['categories'].isin(['Service', 'Service,Financial,Banking and Finance', 'Service,Financial', 'Service,Financial,Stock Brokers', 'Service,Financial,Loans and Mortgages',
                                                                                     'Service,Financial,Financial Planning and Investments', 'Service,Financial,Accounting and Bookkeeping',
                                                                                     'Service,Financial,Taxes', 'Service,Insurance'])) | 
                                           (plaid_txn_debit['merchant_name'].isin(['Internal Revenue Service',
                                                'IRS','Irs Usataxpymt','Franchise Tax Board', 'Irs'
                                                'Withheld','Intuit','Franchise Tax Bo','Federal Tax',
                                                'GEICO','State Farm','Progressive Insurance','Liberty Mutual',
                                                'Allstate','Primerica','Nationwide','Lemonade Insurance',
                                                'Transamerica','Northwestern Mutual']))]

grp7 = df_service_finance.groupby('business_id').agg(total_service_finance_l6m = ('amount', np.sum), avg_txn_amount_service_finance_l6m = ('amount',np.mean), count_txn_service_finance_l6m = ('transaction_id', 'count')).reset_index()

df = pd.merge(df, grp7, how='left', on='business_id')

# # 8. Gambling

# df_gambling = plaid_txn_debit[plaid_txn_debit['merchant_name'].isin(['BetMGM','Betfair','Pulsz',
#                                     'Golden Nugget','LuckyLand Slots','Bet365','UniBet','Performance Predictions',
#                                     'Bingo Cash','Everi','Clearwater Casino'])]

# grp8 = df_gambling.groupby('business_id').agg(total_gambling_merc_l6m = ('amount', np.sum), avg_txn_amount_gambling_merc_l6m = ('amount',np.mean), count_txn_gambling_merc_l6m = ('transaction_id', 'count')).reset_index()

# df = pd.merge(df, grp8, how='left', on='business_id')



## Repeat the same for 1 month

plaid_txn_debit  = plaid_txn_l1m


# 1. Payroll Spend

df_pr = plaid_txn_debit[(plaid_txn_debit['categories'] == 'Transfer,Payroll') | (plaid_txn_debit['merchant_name'].isin(['ADP', 'Gusto', 'Paychex Tps', 'paychex', 'Paychex Eib']))]

grp1 = df_pr.groupby('business_id').agg(total_payroll_spend_l1m = ('amount', np.sum), avg_txn_amount_payroll_spend_l1m = ('amount',np.mean), txn_count_payroll_spend_l1m = ('transaction_id', 'count')).reset_index()

df = pd.merge(df, grp1, how='left', on='business_id')


## 2. Rent

df_rent = plaid_txn_debit[plaid_txn_debit['categories'] == 'Payment,Rent']

grp2 = df_rent.groupby('business_id').agg(total_rent_spend_l1m = ('amount', np.sum), avg_txn_amount_rent_spend_l1m = ('amount',np.mean), txn_count_rent_spend_l1m = ('transaction_id', 'count')).reset_index()

df = pd.merge(df, grp2, how='left', on='business_id')



# 3. Utility bills

df_utility = plaid_txn_debit[(plaid_txn_debit['categories'].isin(['Service,Telecommunication Services', 
                                                                             'Service,Subscription',
                                                                             'Service,Utilities,Electric', 
                                                                             'Service,Utilities', 'Transfer,Billpay', 
                                                                             'Service,Utilities,Water', 
                                                                             'Service,Internet Services'])) 
                                   | (plaid_txn_debit['merchant_name'].isin(['Dominion Energy',
                                    'So Cal Gas','Xcel Energy','FPL','Eversource','National Grid',
                                    'Dukeenergy','PG&E','Spire','Atmos Energy','Southwest Gas',
                                    'FirstEnergy','NV Energy','So Cal Edison','DTE Energy']))]

grp3 = df_utility.groupby('business_id').agg(total_utility_spend_l1m = ('amount', np.sum), avg_txn_amount_utility_spend_l1m = ('amount',np.mean), count_txn_utility_spend_l1m = ('transaction_id', 'count')).reset_index()

df = pd.merge(df, grp3, how='left', on='business_id')


# 4. Marketing Spend

df_ad_spend = plaid_txn_debit[plaid_txn_debit['categories'].isin(['Service,Advertising and Marketing,Online Advertising', 'Service,Advertising and Marketing'])]

grp4 = df_ad_spend.groupby('business_id').agg(total_ad_spend_l1m = ('amount', np.sum), avg_txn_amount_ads_spend_l1m = ('amount',np.mean), count_txn_ads_spend_l1m = ('transaction_id', 'count')).reset_index()

df = pd.merge(df, grp4, how='left', on='business_id')



# 5. Credit card payment

df_cc_payment = plaid_txn_debit[plaid_txn_debit['categories'].isin(['Payment,Credit Card'])]

grp5 = df_cc_payment.groupby('business_id').agg(total_cc_payment_l1m = ('amount', np.sum), avg_txn_amount_cc_payment_l1m = ('amount',np.mean), count_txn_cc_payment_l1m = ('transaction_id', 'count')).reset_index()

df = pd.merge(df, grp5, how='left', on='business_id')



# 6. Personal spend

df_personal_spend = plaid_txn_debit[plaid_txn_debit['categories'].isin(['Shops,Supermarkets and Groceries', 'Shops,Sporting Goods',
                    'Travel,Gas Stations', 'Food and Drink,Restaurants', 'Shops,Digital Purchase', 'Shops,Computers and Electronics',
                    'Shops,Discount Stores', 'Travel,Parking', 'Service,Home Improvement,Swimming Pool Maintenance and Services', 'Shops,Bookstores',
                    'Shops,Warehouses and Wholesale Stores', 'Food and Drink,Restaurants,Fast Food',
                    'Shops,Jewelry and Watches', 'Food and Drink', 'Food and Drink,Restaurants,Coffee Shop', 'Shops,Food and Beverage Store', 'Service,Food and Beverage', 'Shops'])]


grp6 = df_personal_spend.groupby('business_id').agg(total_personal_spend_l1m = ('amount', np.sum), avg_txn_amount_personal_spend_l1m = ('amount',np.mean), count_txn_personal_spend_l1m = ('transaction_id', 'count')).reset_index()

df = pd.merge(df, grp6, how='left', on='business_id')


# 7. Service finance

df_service_finance = plaid_txn_debit[(plaid_txn_debit['categories'].isin(['Service', 'Service,Financial,Banking and Finance', 'Service,Financial', 'Service,Financial,Stock Brokers', 'Service,Financial,Loans and Mortgages',
                                                                                     'Service,Financial,Financial Planning and Investments', 'Service,Financial,Accounting and Bookkeeping',
                                                                                     'Service,Financial,Taxes', 'Service,Insurance'])) | 
                                           (plaid_txn_debit['merchant_name'].isin(['Internal Revenue Service',
                                                'IRS','Irs Usataxpymt','Franchise Tax Board', 'Irs'
                                                'Withheld','Intuit','Franchise Tax Bo','Federal Tax',
                                                'GEICO','State Farm','Progressive Insurance','Liberty Mutual',
                                                'Allstate','Primerica','Nationwide','Lemonade Insurance',
                                                'Transamerica','Northwestern Mutual']))]

grp7 = df_service_finance.groupby('business_id').agg(total_service_finance_l1m = ('amount', np.sum), avg_txn_amount_service_finance_l1m = ('amount',np.mean), count_txn_service_finance_l1m = ('transaction_id', 'count')).reset_index()

df = pd.merge(df, grp7, how='left', on='business_id')


# # 8. Gambling

# df_gambling = plaid_txn_debit[plaid_txn_debit['merchant_name'].isin(['BetMGM','Betfair','Pulsz',
#                                     'Golden Nugget','LuckyLand Slots','Bet365','UniBet','Performance Predictions',
#                                     'Bingo Cash','Everi','Clearwater Casino'])]

# grp8 = df_gambling.groupby('business_id').agg(total_gambling_merc_l1m = ('amount', np.sum), avg_txn_amount_gambling_merc_l1m = ('amount',np.mean), count_txn_gambling_merc_l1m = ('transaction_id', 'count')).reset_index()

# df = pd.merge(df, grp8, how='left', on='business_id')


df.fillna(0, inplace=True)

### Average variables

df['avg_payroll_l3m'] = df['total_payroll_spend_l3m'] / 3
df['avg_payroll_l6m'] = df['total_payroll_spend_l6m'] / 6

df['avg_rent_l3m'] = df['total_rent_spend_l6m'] / 3
df['avg_rent_l6m'] = df['total_rent_spend_l6m'] / 6

df['avg_utility_l3m'] = df['total_utility_spend_l3m'] / 3
df['avg_utility_l6m'] = df['total_utility_spend_l6m'] / 6

df['avg_ad_spend_l3m'] = df['total_ad_spend_l3m'] / 3
df['avg_ad_spend_l6m'] = df['total_ad_spend_l6m'] / 6

df['avg_cc_payment_l3m'] = df['total_cc_payment_l3m'] / 3
df['avg_cc_payment_l6m'] = df['total_cc_payment_l6m'] / 6

df['avg_personal_spend_l3m'] = df['total_personal_spend_l3m'] / 3
df['avg_personal_spend_l6m'] = df['total_personal_spend_l6m'] / 6

df['avg_service_finance_l3m'] = df['total_service_finance_l3m'] / 3
df['avg_service_finance_l6m'] = df['total_service_finance_l6m'] / 6

# df['avg_gambling_merc_l3m'] = df['total_gambling_merc_l3m'] / 3
# df['avg_gambling_merc_l6m'] = df['total_gambling_merc_l6m'] / 6


df.fillna(0, inplace=True)

## Ratio / trend variables

df['trend_payroll_l3l6m'] = df['avg_payroll_l3m'] / df['avg_payroll_l6m'] 

df['trend_rent_l3l6m'] = df['avg_rent_l3m'] / df['avg_rent_l6m'] 

df['trend_utility_l3l6m'] = df['avg_utility_l3m'] / df['avg_utility_l6m'] 

df['trend_ad_spend_l3l6m'] = df['avg_ad_spend_l3m'] / df['avg_ad_spend_l6m'] 

df['trend_cc_payment_l3l6m'] = df['avg_cc_payment_l3m'] / df['avg_cc_payment_l6m'] 

df['trend_personal_spend_l3l6m'] = df['avg_personal_spend_l3m'] / df['avg_personal_spend_l6m'] 

df['trend_service_finance_l3l6m'] = df['avg_service_finance_l3m'] / df['avg_service_finance_l6m'] 

# df['trend_gambling_merc_l6l3m'] = df['avg_gambling_merc_l6m'] / df['avg_gambling_merc_l3m'] 


# derive 1m to 6m ratio

df['trend_payroll_l1l6m'] = df['total_payroll_spend_l1m'] / df['avg_payroll_l6m']

df['trend_rent_l1l6m'] = df['total_rent_spend_l1m'] / df['avg_rent_l6m']

df['trend_utility_l1l6m'] = df['total_utility_spend_l1m'] / df['avg_utility_l6m']

df['trend_ad_spend_l1l6m'] = df['total_ad_spend_l1m'] / df['avg_ad_spend_l6m']

df['trend_cc_payment_l1l6m'] = df['total_cc_payment_l1m'] / df['avg_cc_payment_l6m']

df['trend_personal_spend_l1l6m'] = df['total_personal_spend_l1m'] / df['avg_personal_spend_l6m']

df['trend_service_finance_l1l6m'] = df['total_service_finance_l1m'] / df['avg_service_finance_l6m']

# df['trend_gambling_merc_l6l1m'] = df['avg_gambling_merc_l6m'] / df['total_gambling_merc_l1m'] 



## total debit


total_debit_l1m = plaid_txn_l1m.groupby('business_id').agg(total_debit_amount_l1m = ('amount', np.sum)).reset_index()

df = pd.merge(df, total_debit_l1m, how='left', on='business_id')

total_debit_l3m = plaid_txn_l3m.groupby('business_id').agg(total_debit_amount_l3m = ('amount', np.sum)).reset_index()

df = pd.merge(df, total_debit_l3m, how='left', on='business_id')

total_debit_l6m =  plaid_txn_l6m.groupby('business_id').agg(total_debit_amount_l6m = ('amount', np.sum)).reset_index()

df = pd.merge(df, total_debit_l6m, how='left', on='business_id')


df.fillna(0, inplace=True)

# ratio

df['perc_payroll_spend_l6m'] = (df['total_payroll_spend_l6m'] / df['total_debit_amount_l6m']) * 100

df['perc_rent_spend_l6m'] = (df['total_rent_spend_l6m'] / df['total_debit_amount_l6m']) * 100

df['perc_utility_spend_l6m'] = (df['total_utility_spend_l6m'] / df['total_debit_amount_l6m']) * 100

df['perc_ad_spend_l6m'] = (df['total_ad_spend_l6m'] / df['total_debit_amount_l6m']) * 100

df['perc_cc_payment_spend_l6m'] = (df['total_cc_payment_l6m'] / df['total_debit_amount_l6m']) * 100

df['perc_personal_spend_l6m'] = (df['total_personal_spend_l6m'] / df['total_debit_amount_l6m']) * 100

df['perc_service_finance_spend_l6m'] = (df['total_service_finance_l6m'] / df['total_debit_amount_l6m']) * 100

# df['perc_gambling_merc_spend_l6m'] = (df['total_gambling_merc_l6m'] / df['total_debit_amount_l6m']) * 100

# 
df['perc_payroll_spend_l3m'] = (df['total_payroll_spend_l3m'] / df['total_debit_amount_l3m']) * 100

df['perc_rent_spend_l3m'] = (df['total_rent_spend_l3m'] / df['total_debit_amount_l3m']) * 100

df['perc_utility_spend_l3m'] = (df['total_utility_spend_l3m'] / df['total_debit_amount_l3m']) * 100

df['perc_ad_spend_l3m'] = (df['total_ad_spend_l3m'] / df['total_debit_amount_l3m']) * 100

df['perc_cc_payment_spend_l3m'] = (df['total_cc_payment_l3m'] / df['total_debit_amount_l3m']) * 100

df['perc_personal_spend_l3m'] = (df['total_personal_spend_l3m'] / df['total_debit_amount_l3m']) * 100

df['perc_service_finance_spend_l3m'] = (df['total_service_finance_l3m'] / df['total_debit_amount_l3m']) * 100

# df['perc_gambling_merc_spend_l3m'] = (df['total_gambling_merc_l3m'] / df['total_debit_amount_l3m']) * 100


## counts of txns greater than 500 dollars

# invoice of 



        
## Infinity and missing value 

df.replace({np.inf:0, -np.inf:0}, inplace=True)

df = df.fillna(0)

df_main = df.copy()

df_raw = df_main.copy()

## remove non-transactional features

df_raw = df_raw.drop(['business_id', 'lending_business_id','decision_date','drawn_flag', 'everDPD_15', 'fico_score'], axis=1)






## train-test split

# train test split
x_train, y_train, x_test, y_test = split_test_train(df_raw, target_column='target', test_size=0.3, random_state=seed)
print(f'{x_train.shape = }', '|' ,f'{y_train.shape = }', '|' ,f'{x_test.shape = }', '|' ,f'{y_test.shape = }')


# copy to df
df = x_train.copy(deep=True)

# get datatypes frequency
def get_datatypes_freq(df):
    type_dct = {str(k): list(v) for k, v in df.groupby(df.dtypes, axis=1)}
    type_dct_info = {k: len(v) for k, v in type_dct.items()}
    return type_dct, type_dct_info

type_dct, type_dct_info = get_datatypes_freq(df)
type_dct_info


# get constant features
def get_const_features(df):
    const_list = []
    for col in df.columns: 
        if (len(df[col].unique())==1):
            const_list.append(col)
    return(const_list)

# remove constant features
const_list = get_const_features(df)
df = df.drop(columns=const_list)
df.shape


# get quasi-constant features
def get_quasi_const_features(df, threshold=0.01):
    qconst_list = []
    for col in df.columns: 
        if (df[col].var() <= threshold):
            qconst_list.append(col)
    return(qconst_list)

# remove constant features
qconst_list = get_quasi_const_features(df, threshold=0.01)
df = df.drop(columns=qconst_list)
df.shape


# view missing values
def missing_value(df):
    percent_missing = df.isnull().sum() * 100 / len(df)
    missing_val_df = pd.DataFrame({'percent_missing': percent_missing})
    missing_val_df.sort_values(by='percent_missing', ascending=False, inplace=True)
    return missing_val_df

missing_value(df)



# get boolean columns
def findbool(df):
    bool_arr = []
    for col in df.columns: 
        if (len(df[col].unique())<=2):
            bool_arr.append(col)
    return(bool_arr)

bool_col_list = findbool(df)
len(bool_col_list)

type_dct, type_dct_info = get_datatypes_freq(df)
col_list = (type_dct['float64'])
col_list_excpt_bool = [column for column in col_list if column not in bool_col_list]
len(col_list_excpt_bool)

## iv_woe

def iv_woe(data, target, bins=10, show_woe=False):
    
    #Empty Dataframe
    newDF,woeDF = pd.DataFrame(), pd.DataFrame()
    
    #Extract Column Names
    cols = data.columns
    
    #Run WOE and IV on all the independent variables
    for ivars in cols[~cols.isin([target])]:
        if (data[ivars].dtype.kind in 'bifc') and (len(np.unique(data[ivars]))>10):
            binned_x = pd.qcut(data[ivars], bins,  duplicates='drop')
            d0 = pd.DataFrame({'x': binned_x, 'y': data[target]})
        else:
            d0 = pd.DataFrame({'x': data[ivars], 'y': data[target]})

        
        # Calculate the number of events in each group (bin)
        d = d0.groupby("x", as_index=False).agg({"y": ["count", "sum"]})
        d.columns = ['Cutoff', 'N', 'Events']
        
        # Calculate % of events in each group.
        d['% of Events'] = np.maximum(d['Events'], 0.5) / d['Events'].sum()

        # Calculate the non events in each group.
        d['Non-Events'] = d['N'] - d['Events']
        # Calculate % of non events in each group.
        d['% of Non-Events'] = np.maximum(d['Non-Events'], 0.5) / d['Non-Events'].sum()

        # Calculate WOE by taking natural log of division of % of non-events and % of events
        d['WoE'] = np.log(d['% of Events']/d['% of Non-Events'])
        d['IV'] = d['WoE'] * (d['% of Events'] - d['% of Non-Events'])
        d.insert(loc=0, column='Variable', value=ivars)
        print("Information value of " + ivars + " is " + str(round(d['IV'].sum(),6)))
        temp =pd.DataFrame({"Variable" : [ivars], "IV" : [d['IV'].sum()]}, columns = ["Variable", "IV"])
        newDF=pd.concat([newDF,temp], axis=0)
        woeDF=pd.concat([woeDF,d], axis=0)

        #Show WOE Table
        if show_woe == True:
            print(d)
    return newDF, woeDF


# remove features on basis of IV
# y_train.reset_index(drop=True, inplace=True)
df['target'] = y_train
df['target'] = df['target'].astype(float)
temp = df.copy()

t1, t2 = iv_woe(temp[np.append(col_list_excpt_bool,['target'])], 'target', bins=5, show_woe=False)
feature_list = list(t1[ (t1['IV']<iv_upper_limit) & (t1['IV']>iv_lower_limit) ]['Variable'].values)
len(feature_list)



# view correlation
corr_df, subset_df = sel.get_correlated_features(df, feature_list, thresh=corr_arr)
corr_df


# remove correlated features
feature_list = sel.corr_iter(df, np.array(feature_list), thresh=corr_arr)
feature_list = list(feature_list)
len(feature_list)


# get feature list after iterative VIF elimination
def vif_iter(df, iv, threshold=10):
    vif_data = pd.DataFrame()
    vif_data["feature"] = iv
    vif_data["VIF"] = [variance_inflation_factor(df[iv].values, i) for i in range(len(iv))]
    if len(vif_data[vif_data['VIF'] == np.inf]) > 0:
        feature = vif_data[vif_data['VIF'] == np.inf]['feature'].iloc[0]
        iv.remove(feature)
        vif_iter(df, iv, threshold)
    elif len(vif_data[vif_data['VIF'] > threshold]) > 0:
        feature = vif_data.sort_values(by='VIF', ascending=False)['feature'].iloc[0]
        iv.remove(feature)
        vif_iter(df, iv, threshold)
    vif_data = pd.DataFrame()
    vif_data["feature"] = iv
    vif_data["VIF"] = [variance_inflation_factor(df[iv].values, i) for i in range(len(iv))]
    return iv, vif_data

feature_list, vif_df = vif_iter(df, feature_list, threshold=vif_arr)
len(feature_list)


# Backward feature elimination
feat_list = ft.backward_feature_selection(df[feature_list], y_train, num_features=features_arr)
feat_list

# manual eyeballing 
# remove gambling and additional columns

feat_list = ['avg_txn_amount_service_finance_l3m',
                 'txn_count_payroll_spend_l1m',
                 'txn_count_rent_spend_l1m',
                 'count_txn_utility_spend_l1m',
                 'avg_txn_amount_cc_payment_l1m',
                 'count_txn_cc_payment_l1m',
                 'total_personal_spend_l1m',
                 'count_txn_personal_spend_l1m',
                 'avg_txn_amount_service_finance_l1m',
                 'count_txn_service_finance_l1m',
                 'trend_utility_l1l6m',
                 'trend_ad_spend_l1l6m',
                 'trend_personal_spend_l1l6m',
                 'perc_utility_spend_l3m',
                 'perc_service_finance_spend_l3m']



## think about sequence here--- bfe vs. woe binning


## optimal binning woe

import optbinning as optb
from optbinning import Scorecard, BinningProcess, OptimalBinning
from optbinning.binning.binning_statistics import BinningTable

df_temp = df[feat_list].copy()
df_temp['target'] = df['target']


# 1
Xt= df_temp['avg_txn_amount_service_finance_l3m']
yt = df_temp['target'].astype(int)

optb = OptimalBinning(name='avg_txn_amount_service_finance_l3m', dtype="numerical", max_n_prebins=4, monotonic_trend='descending')
optb.fit(Xt, yt)  

Xt_binned = optb.transform(Xt)

ob_avg_txn_amount_service_finance_l3m = optb.binning_table.build()

optb.binning_table.plot(metric="event_rate")



# Var tranform
transformed_vars = x_train[feat_list]

# transform
col         = 'avg_txn_amount_service_finance_l3m'
conditions  = [ transformed_vars[col] <= 287, 
                transformed_vars[col] > 287 ]

choices     = [-0.128497, 0.431184]
    
transformed_vars["avg_txn_amount_service_finance_l3m"] = np.select(conditions, choices, default=np.nan)

# 2
Xt= df_temp['txn_count_payroll_spend_l1m']
yt = df_temp['target'].astype(int)

optb = OptimalBinning(name='txn_count_payroll_spend_l1m', dtype="numerical", max_n_prebins=4, special_codes=[0], monotonic_trend='descending')
optb.fit(Xt, yt)

Xt_binned = optb.transform(Xt)

ob_txn_count_payroll_spend_l1m = optb.binning_table.build()
optb.binning_table.plot(metric="event_rate")


# transform
col         = 'txn_count_payroll_spend_l1m'
conditions  = [ transformed_vars[col] <= 0, 
                transformed_vars[col] > 0 ]

choices     = [-0.00409379, 0.115614]
    
transformed_vars["txn_count_payroll_spend_l1m"] = np.select(conditions, choices, default=np.nan)

# 3
Xt= df_temp['txn_count_rent_spend_l1m']
yt = df_temp['target'].astype(int)

optb = OptimalBinning(name='txn_count_rent_spend_l1m', dtype="numerical", max_n_prebins=4, special_codes=[0], monotonic_trend='descending')
optb.fit(Xt, yt)

Xt_binned = optb.transform(Xt)

ob_txn_count_rent_spend_l1m = optb.binning_table.build()
optb.binning_table.plot(metric="event_rate")


# transform
col         = 'txn_count_rent_spend_l1m'
conditions  = [ transformed_vars[col] <= 0, 
                transformed_vars[col] > 0 ]

choices     = [0.0723854, -0.619835]
    
transformed_vars["txn_count_rent_spend_l1m"] = np.select(conditions, choices, default=np.nan)

# 4
Xt= df_temp['count_txn_utility_spend_l1m']
yt = df_temp['target'].astype(int)

optb = OptimalBinning(name='count_txn_utility_spend_l1m', dtype="numerical", max_n_prebins=4, monotonic_trend='ascending')
optb.fit(Xt, yt)

Xt_binned = optb.transform(Xt)

ob_count_txn_utility_spend_l1m = optb.binning_table.build()
optb.binning_table.plot(metric="event_rate")


# transform
col         = 'count_txn_utility_spend_l1m'
conditions  = [ transformed_vars[col] <= 0, 
                (transformed_vars[col] > 0) & (transformed_vars[col]<= 2.5),
                (transformed_vars[col] > 2.5)]

choices     = [0.085333, 0.0961961, -0.41463]

    
transformed_vars["count_txn_utility_spend_l1m"] = np.select(conditions, choices, default=np.nan)


# 5
Xt= df_temp['avg_txn_amount_cc_payment_l1m']
yt = df_temp['target'].astype(int)

optb = OptimalBinning(name='avg_txn_amount_cc_payment_l1m', dtype="numerical", max_n_prebins=4, monotonic_trend='descending')
optb.fit(Xt, yt)

Xt_binned = optb.transform(Xt)

ob_avg_txn_amount_cc_payment_l1m = optb.binning_table.build()
optb.binning_table.plot(metric="event_rate")


# transform
col         = 'avg_txn_amount_cc_payment_l1m'
conditions  = [ transformed_vars[col] <= 278, 
                (transformed_vars[col] > 278) & (transformed_vars[col]<= 891),
                (transformed_vars[col] > 891)]

choices     = [-0.338, 0.555447, 2.77641]

    
transformed_vars["avg_txn_amount_cc_payment_l1m"] = np.select(conditions, choices, default=np.nan)



# # 6
# Xt= df_temp['count_txn_cc_payment_l1m']
# yt = df_temp['target'].astype(int)

# optb = OptimalBinning(name='count_txn_cc_payment_l1m', dtype="numerical", max_n_prebins=4, monotonic_trend='descending')
# optb.fit(Xt, yt)

# Xt_binned = optb.transform(Xt)

# ob_count_txn_cc_payment_l1m = optb.binning_table.build()
# optb.binning_table.plot(metric="event_rate")


# # transform
# col         = 'count_txn_cc_payment_l1m'
# conditions  = [ transformed_vars[col] <= 0, 
#                 (transformed_vars[col] > 0) & (transformed_vars[col]<= 6),
#                 (transformed_vars[col] > 6)]

# choices     = [-0.252111, -0.0686323, 0.980033]

    
# transformed_vars["count_txn_cc_payment_l1m"] = np.select(conditions, choices, default=np.nan)


# 7
Xt= df_temp['total_personal_spend_l1m']
yt = df_temp['target'].astype(int)

optb = OptimalBinning(name='total_personal_spend_l1m', dtype="numerical", max_n_prebins=4, monotonic_trend='ascending')
optb.fit(Xt, yt)

Xt_binned = optb.transform(Xt)

ob_total_personal_spend_l1m = optb.binning_table.build()
optb.binning_table.plot(metric="event_rate")


# transform
col         = 'total_personal_spend_l1m'
conditions  = [ transformed_vars[col] <= 381, 
                (transformed_vars[col] > 381)]

choices     = [0.354025, -0.430518]

    
transformed_vars["total_personal_spend_l1m"] = np.select(conditions, choices, default=np.nan)

# 8
Xt= df_temp['count_txn_personal_spend_l1m']
yt = df_temp['target'].astype(int)

optb = OptimalBinning(name='count_txn_personal_spend_l1m', dtype="numerical", max_n_prebins=4, monotonic_trend='ascending')
optb.fit(Xt, yt)

Xt_binned = optb.transform(Xt)

ob_count_txn_personal_spend_l1m = optb.binning_table.build()
optb.binning_table.plot(metric="event_rate")


# transform
col         = 'count_txn_personal_spend_l1m'
conditions  = [ transformed_vars[col] <= 3, 
               (transformed_vars[col] > 3) & (transformed_vars[col]<= 13),
                (transformed_vars[col] > 13)]

choices     = [0.418257, 0.0561908, -0.602313]

    
transformed_vars["count_txn_personal_spend_l1m"] = np.select(conditions, choices, default=np.nan)


# # 9
# Xt= df_temp['avg_txn_amount_service_finance_l1m']
# yt = df_temp['target'].astype(int)

# optb = OptimalBinning(name='avg_txn_amount_service_finance_l1m', dtype="numerical", max_n_prebins=4, monotonic_trend='descending')
# optb.fit(Xt, yt) 

# Xt_binned = optb.transform(Xt)

# ob_avg_txn_amount_service_finance_l1m = optb.binning_table.build()
# optb.binning_table.plot(metric="event_rate")


# # transform
# col         = 'avg_txn_amount_service_finance_l1m'
# conditions  = [ transformed_vars[col] <= 101, 
#                 (transformed_vars[col] > 101)]

# choices     = [-0.152738, 0.311161]

    
# transformed_vars["avg_txn_amount_service_finance_l1m"] = np.select(conditions, choices, default=np.nan)

# # 10
# Xt= df_temp['count_txn_service_finance_l1m']
# yt = df_temp['target'].astype(int)

# optb = OptimalBinning(name='count_txn_service_finance_l1m', dtype="numerical", max_n_prebins=4, monotonic_trend='ascending')
# optb.fit(Xt, yt)

# Xt_binned = optb.transform(Xt)

# ob_count_txn_service_finance_l1m = optb.binning_table.build()
# optb.binning_table.plot(metric="event_rate")


# # transform
# col         = 'count_txn_service_finance_l1m'
# conditions  = [ transformed_vars[col] <= 4,
#                (transformed_vars[col] > 4) & (transformed_vars[col]<= 10),
#                 (transformed_vars[col] > 10)]

# choices     = [0.168061, -0.287829, -0.588583]

    
# transformed_vars["count_txn_service_finance_l1m"] = np.select(conditions, choices, default=np.nan)

# 11
Xt= df_temp['trend_utility_l1l6m']
yt = df_temp['target'].astype(int)

optb = OptimalBinning(name='trend_utility_l1l6m', dtype="numerical", max_n_prebins=4, monotonic_trend='ascending')
optb.fit(Xt, yt)

Xt_binned = optb.transform(Xt)

ob_trend_utility_l1l6m = optb.binning_table.build()
optb.binning_table.plot(metric="event_rate")


# transform
col         = 'trend_utility_l1l6m'
conditions  = [ transformed_vars[col] <= 1.10,
                (transformed_vars[col] > 1.10)]

choices     = [0.14527, -0.596951]

    
transformed_vars["trend_utility_l1l6m"] = np.select(conditions, choices, default=np.nan)

# 12
Xt= df_temp['trend_ad_spend_l1l6m']
yt = df_temp['target'].astype(int)

optb = OptimalBinning(name='trend_ad_spend_l1l6m', dtype="numerical", max_n_prebins=4, special_codes=[0])
optb.fit(Xt, yt)

Xt_binned = optb.transform(Xt)

ob_trend_ad_spend_l1l6m = optb.binning_table.build()
optb.binning_table.plot(metric="event_rate")


# transform
col         = 'trend_ad_spend_l1l6m'
conditions  = [ transformed_vars[col] <= 0,
                (transformed_vars[col] > 0)]

choices     = [0.0355715, -0.000393683]

    
transformed_vars["trend_ad_spend_l1l6m"] = np.select(conditions, choices, default=np.nan)


# 13
Xt= df_temp['trend_personal_spend_l1l6m']
yt = df_temp['target'].astype(int)

optb = OptimalBinning(name='trend_personal_spend_l1l6m', dtype="numerical", max_n_prebins=4, monotonic_trend='ascending')
optb.fit(Xt, yt)

Xt_binned = optb.transform(Xt)

ob_trend_personal_spend_l1l6m = optb.binning_table.build()
optb.binning_table.plot(metric="event_rate")


# transform
col         = 'trend_personal_spend_l1l6m'
conditions  = [ transformed_vars[col] <= 0.6,
                (transformed_vars[col] > 0.6)]

choices     = [0.313027, -0.379944]

    
transformed_vars["trend_personal_spend_l1l6m"] = np.select(conditions, choices, default=np.nan)

# 14
Xt= df_temp['perc_utility_spend_l3m']
yt = df_temp['target'].astype(int)

optb = OptimalBinning(name='perc_utility_spend_l3m', dtype="numerical", max_n_prebins=4, monotonic_trend='ascending')
optb.fit(Xt, yt)

Xt_binned = optb.transform(Xt)

ob_perc_utility_spend_l3m = optb.binning_table.build()
optb.binning_table.plot(metric="event_rate")


# transform
col         = 'perc_utility_spend_l3m'
conditions  = [ transformed_vars[col] <= 0.22,
               (transformed_vars[col] > 0.22) & (transformed_vars[col]<= 8.65),
                (transformed_vars[col] > 8.65)]

choices     = [0.32805, -0.289015, -0.596951]

transformed_vars["perc_utility_spend_l3m"] = np.select(conditions, choices, default=np.nan)

# 15
Xt= df_temp['perc_service_finance_spend_l3m']
yt = df_temp['target'].astype(int)

optb = OptimalBinning(name='perc_service_finance_spend_l3m', dtype="numerical", max_n_prebins=4, monotonic_trend='ascending')
optb.fit(Xt, yt)

Xt_binned = optb.transform(Xt)

ob_perc_service_finance_spend_l3m = optb.binning_table.build()
optb.binning_table.plot(metric="event_rate")


# transform
col         = 'perc_service_finance_spend_l3m'
conditions  = [ transformed_vars[col] <= 5.4,
               (transformed_vars[col] > 5.4) & (transformed_vars[col]<= 23.1),
                (transformed_vars[col] > 23.1)]

choices     = [0.254261, -0.198043, -0.552215]

transformed_vars["perc_service_finance_spend_l3m"] = np.select(conditions, choices, default=np.nan)






# copy vars
X_train = transformed_vars.copy()

# hyperparameters
params_log_reg = {'penalty': 'l2',
                  'random_state': seed,
                  'solver': 'liblinear',
                  'class_weight': 'balanced'}

# model fit
logreg_model = classification_models(X_train[feat_list], y_train, params_log_reg, models=['log_reg'])

# train cv scores
cv_scores = cross_validation(logreg_model, X_train[feat_list], y_train, scoring='roc_auc', folds=3, seed=seed)
print('CV Scores -',np.round(cv_scores, 2))
print('Mean of CV Scores -',np.round(np.mean(cv_scores),2))

# train score
# model_metrics(logreg_model.predict(transformed_vars[feat_list]), np.array(y_train), logreg_model.predict_proba(transformed_vars[feat_list]))


# Feature importance
feat_imp = feature_importance(logreg_model, X_train[feat_list], show_plot=True)

feat_imp.sort_values(by='importance', ascending=False)





## Test model


# reset index
# x_test.reset_index(drop=True, inplace=True)

# binning variable transform
x_test.fillna(0, inplace=True)

transformed_vars = x_test[feat_list]

# transform
col         = 'avg_txn_amount_service_finance_l3m'
conditions  = [ transformed_vars[col] <= 287, 
                transformed_vars[col] > 287 ]

choices     = [-0.128497, 0.431184]
    
transformed_vars["avg_txn_amount_service_finance_l3m"] = np.select(conditions, choices, default=np.nan)


col         = 'txn_count_payroll_spend_l1m'
conditions  = [ transformed_vars[col] <= 0, 
                transformed_vars[col] > 0 ]

choices     = [-0.00409379, 0.115614]
    
transformed_vars["txn_count_payroll_spend_l1m"] = np.select(conditions, choices, default=np.nan)

# transform
col         = 'txn_count_rent_spend_l1m'
conditions  = [ transformed_vars[col] <= 0, 
                transformed_vars[col] > 0 ]

choices     = [0.0723854, -0.619835]
    
transformed_vars["txn_count_rent_spend_l1m"] = np.select(conditions, choices, default=np.nan)

col         = 'count_txn_utility_spend_l1m'
conditions  = [ transformed_vars[col] <= 0, 
                (transformed_vars[col] > 0) & (transformed_vars[col]<= 2.5),
                (transformed_vars[col] > 2.5)]

choices     = [0.085333, 0.0961961, -0.41463]

    
transformed_vars["count_txn_utility_spend_l1m"] = np.select(conditions, choices, default=np.nan)

col         = 'avg_txn_amount_cc_payment_l1m'
conditions  = [ transformed_vars[col] <= 278, 
                (transformed_vars[col] > 278) & (transformed_vars[col]<= 891),
                (transformed_vars[col] > 891)]

choices     = [-0.338, 0.555447, 2.77641]

    
transformed_vars["avg_txn_amount_cc_payment_l1m"] = np.select(conditions, choices, default=np.nan)


# col         = 'count_txn_cc_payment_l1m'
# conditions  = [ transformed_vars[col] <= 0, 
#                 (transformed_vars[col] > 0) & (transformed_vars[col]<= 6),
#                 (transformed_vars[col] > 6)]

# choices     = [-0.252111, -0.0686323, 0.980033]

    
# transformed_vars["count_txn_cc_payment_l1m"] = np.select(conditions, choices, default=np.nan)


col         = 'total_personal_spend_l1m'
conditions  = [ transformed_vars[col] <= 381, 
                (transformed_vars[col] > 381)]

choices     = [0.354025, -0.430518]

    
transformed_vars["total_personal_spend_l1m"] = np.select(conditions, choices, default=np.nan)

col         = 'count_txn_personal_spend_l1m'
conditions  = [ transformed_vars[col] <= 3, 
               (transformed_vars[col] > 3) & (transformed_vars[col]<= 13),
                (transformed_vars[col] > 13)]

choices     = [0.418257, 0.0561908, -0.602313]

    
transformed_vars["count_txn_personal_spend_l1m"] = np.select(conditions, choices, default=np.nan)

# col         = 'avg_txn_amount_service_finance_l1m'
# conditions  = [ transformed_vars[col] <= 101, 
#                 (transformed_vars[col] > 101)]

# choices     = [-0.152738, 0.311161]

    
# transformed_vars["avg_txn_amount_service_finance_l1m"] = np.select(conditions, choices, default=np.nan)

# col         = 'count_txn_service_finance_l1m'
# conditions  = [ transformed_vars[col] <= 4,
#                (transformed_vars[col] > 4) & (transformed_vars[col]<= 10),
#                 (transformed_vars[col] > 10)]

# choices     = [0.168061, -0.287829, -0.588583]

    
# transformed_vars["count_txn_service_finance_l1m"] = np.select(conditions, choices, default=np.nan)

col         = 'trend_utility_l1l6m'
conditions  = [ transformed_vars[col] <= 1.10,
                (transformed_vars[col] > 1.10)]

choices     = [0.14527, -0.596951]

    
transformed_vars["trend_utility_l1l6m"] = np.select(conditions, choices, default=np.nan)


col         = 'trend_ad_spend_l1l6m'
conditions  = [ transformed_vars[col] <= 0,
                (transformed_vars[col] > 0)]

choices     = [0.0355715, -0.000393683]

    
transformed_vars["trend_ad_spend_l1l6m"] = np.select(conditions, choices, default=np.nan)


col         = 'trend_personal_spend_l1l6m'
conditions  = [ transformed_vars[col] <= 0.6,
                (transformed_vars[col] > 0.6)]

choices     = [0.313027, -0.379944]

    
transformed_vars["trend_personal_spend_l1l6m"] = np.select(conditions, choices, default=np.nan)


col         = 'perc_utility_spend_l3m'
conditions  = [ transformed_vars[col] <= 0.22,
               (transformed_vars[col] > 0.22) & (transformed_vars[col]<= 8.65),
                (transformed_vars[col] > 8.65)]

choices     = [0.32805, -0.289015, -0.596951]

transformed_vars["perc_utility_spend_l3m"] = np.select(conditions, choices, default=np.nan)

col         = 'perc_service_finance_spend_l3m'
conditions  = [ transformed_vars[col] <= 5.4,
               (transformed_vars[col] > 5.4) & (transformed_vars[col]<= 23.1),
                (transformed_vars[col] > 23.1)]

choices     = [0.254261, -0.198043, -0.552215]

transformed_vars["perc_service_finance_spend_l3m"] = np.select(conditions, choices, default=np.nan)




# copy
X_test = transformed_vars.copy()


# test cv scores
cv_scores = cross_validation(logreg_model, X_test[feat_list], y_test, scoring='roc_auc', folds=3, seed=seed)
print('CV Scores -',np.round(cv_scores, 2))
print('Mean of CV Scores -',np.round(np.mean(cv_scores),2))


# test score
# model_metrics(logreg_model.predict(X_test[feat_list]), np.array(y_test), logreg_model.predict_proba(X_test[feat_list]))



## Model Evaluation - KS & ROC AUC

def ks(target=None, prob=None):
    data = pd.DataFrame()
    data['y'] = target
    data['y'] = data['y'].astype(float)
    data['p'] = prob
    data['y0'] = 1- data['y']
    data['bucket'] = pd.qcut(data['p'], 5)
    grouped = data.groupby('bucket', as_index=False)
    kstable = pd.DataFrame()
    kstable['min_prob'] = grouped.min()['p']
    kstable['max_prob'] = grouped.max()['p']
    kstable['events'] = grouped.sum()['y']
    kstable['nonevents'] = grouped.sum()['y0']
    kstable = kstable.sort_values(by='min_prob', ascending=False).reset_index(drop=True)
    kstable['event_rate'] = (kstable.events / data['y'].sum()).apply('{0:.2%}'.format)
    kstable['nonevent_rate'] = (kstable['nonevents'] /  data['y0'].sum()).apply('{0:2%}'.format)
    kstable['cum_eventrate'] = (kstable.events / data['y'].sum()).cumsum()
    kstable['cum_noneventrate'] = (kstable.nonevents / data['y0'].sum()).cumsum()
    kstable['KS'] = np.round(kstable['cum_eventrate'] - kstable['cum_noneventrate'], 3) * 100
    kstable['bad_rate'] = (kstable['events'] / (kstable['events'] + kstable['nonevents'])) * 100
    
    # formatting
    kstable['cum_eventrate'] = kstable['cum_eventrate'].apply('{0:.2%}'.format)
    kstable['cum_noneventrate'] = kstable['cum_noneventrate'].apply('{0:.2%}'.format)
    kstable.index = range(1,6)
    kstable.index.rename('Decile', inplace=True)
    pd.set_option('display.max_columns', 9)
    print(kstable)
    
    # Display KS
    print("KS is " + str(max(kstable['KS']))+"%"+ " at decile " + str((kstable.index[kstable['KS']==max(kstable['KS'])][0])))
    return kstable

 

# predicted proability
train_pred = logreg_model.predict_proba(X_train[feat_list])[:,1]
                                                                     
test_pred = logreg_model.predict_proba(X_test[feat_list])[:,1]


train_ks = ks(y_train, train_pred)
test_ks = ks(y_test, test_pred)

from sklearn.metrics import  roc_auc_score

print(roc_auc_score(y_train, train_pred))   
  
print(roc_auc_score(y_test, test_pred))    



# ### Credit scoring part

# transformed_vars['pred_proba'] = logreg_model.predict_proba(transformed_vars[feat_list])[:,1)

# transformed_vars['odds'] = transformed_vars['pred_proba'] / (1-transformed_vars['pred_proba'])

# transformed_vars['log_odds'] = np.log(transformed_vars['odds'])

# transformed_vars['risk_score'] = 644.2 - (86.6 * transformed_vars['log_odds'])

# # Note: In the above equation, Offset and factor values should be adjusted accordig to PD in KS table


     
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        