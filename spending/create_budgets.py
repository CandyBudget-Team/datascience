import json
import csv

import numpy as np

from math import exp

# import tensorflow as tf

# mediane des depenses

def add_customer_id(filename_in, filename_out, customer_id):
    with open(filename_in, 'r') as csvin:
        readfile = csv.reader(csvin, delimiter=',')
        with open(filename_out, 'w') as csvout:
            writefile = csv.writer(csvout, delimiter=',', lineterminator='\n')
            for row in readfile:
                result = row + [customer_id]
                writefile.writerow(result)

def write_transactions(customer_id,transactions):
    print("writing transactions")
    # open a file for writing
    account_data_csv = open('capital_transactions_' + str(customer_id) + '.csv', 'w')
    # create the csv writer object
    csvwriter = csv.writer(account_data_csv)
    count = 0
    for emp in transactions:
          if count == 0:
                 header = emp.keys()
                 print(header)
                #  csvwriter.writerow(header)
                 count += 1
          csvwriter.writerow(emp.values())
    account_data_csv.close()

    add_customer_id('capital_transactions_' + str(customer_id) + '.csv', 'capital_transactions_augmented_' + str(customer_id) + '.csv', customer_id)


with open('all_transactions.json') as json_file:
    data = json.load(json_file)
    #print data
    #for l in data:
    #    print account_row_id


    # capital_parsed = json.loads(data)

    print('customer_id,customer_ratio_rewards, customer_ratio_payments,credit_utilization,single_score_for_customer ')

    for row_data in data:
        index = 0
        account_data = row_data["account_row_id"]
        customer_data = row_data["customer"]

        # print(account_data)
        for a_customer in customer_data:
            # print (a_customer.keys())

            # print(account_data)

            # print(a_customer["customer_id"])
            # print(a_customer["email"])

            transactions_data = a_customer["transactions"]
            # print(transactions_data.keys())

            # write_transactions(a_customer["customer_id"],transactions_data)


            # # writing to file
            #
            # account_data_csv = open('capital_transactions.csv', 'w')
            # # create the csv writer object
            # csvwriter = csv.writer(account_data_csv)
            # count = 0
            # for emp in transactions_data:
            #       if count == 0:
            #              header = transactions_data.keys()
            #              print(header)
            #              csvwriter.writerow(header)
            #              count += 1
            #       csvwriter.writerow(emp.values())
            # account_data_csv.close()

            ##############

            # print(" ** ")
            # print(transactions_data[0].keys())
            # for a_transaction in transactions_data:
            #     print(a_transaction["amount"])

        account_balance_data = row_data["account_balance"]

        account_spend_limit_data = row_data["account_spend_limit"]

        # rewards
        rewards_data = row_data["rewards"]
        all_rewards_used = [item['total_rewards_used'] for item in rewards_data]
        all_rewards_earned = [item['total_rewards_earned'] for item in rewards_data]
        reward_ar_used = np.array(all_rewards_used)
        reward_ar_earned = np.array(all_rewards_earned)
        # print(str(np.mean(reward_ar_used) / np.mean(reward_ar_earned)) )
        customer_ratio_rewards = np.mean(reward_ar_used) / np.mean(reward_ar_earned)


        total_reward_remaining_data = row_data["total_rewards_remaining"]

        # payments
        payments_data = row_data["payments"]
        # rewards_data = row_data["rewards"]
        all_balance_paid = [item['total_balance_paid'] for item in payments_data]
        all_monthly_balance = [item['total_monthly_balance'] for item in payments_data]
        payment_ar_paid = np.array(all_balance_paid)
        payment_ar_monthly_balance = np.array(all_monthly_balance)
        # print(str(np.mean(payment_ar_paid) / np.mean(payment_ar_monthly_balance)) )
        customer_ratio_payments = np.mean(payment_ar_paid) / np.mean(payment_ar_monthly_balance)

        credit_utilization =  account_balance_data / account_spend_limit_data

        is_original_data = row_data["is_original"]

        single_score_for_customer = customer_ratio_rewards * exp(customer_ratio_payments) / credit_utilization

        # print('customer_id,customer_ratio_rewards, customer_ratio_payments,credit_utilization ')
        print(" %d ,%f , %f,  %f, %f " % (a_customer["customer_id"], customer_ratio_rewards, customer_ratio_payments,credit_utilization,single_score_for_customer))

        # print ("##")
        # if index == 1 :
        #     break

        # index = index +1

        # account balance / account_spend_limit

# sed 1d capital_transactions_augmented_*.csv > merged.csv

    # # open a file for writing
    # account_data_csv = open('capital_transactions.csv', 'w')
    # # create the csv writer object
    # csvwriter = csv.writer(account_data_csv)
    # count = 0
    # for emp in account_data:
    #       if count == 0:
    #              header = emp.keys()
    #              print(header)
    #              csvwriter.writerow(header)
    #              count += 1
    #       csvwriter.writerow(emp.values())
    # account_data_csv.close()
