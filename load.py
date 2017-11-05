import json
import csv

import numpy as np
import tensorflow as tf

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


with open('capital.json') as json_file:
    data = json.load(json_file)
    #print data
    #for l in data:
    #    print account_row_id


    # capital_parsed = json.loads(data)



    for row_data in data:
        index = 0
        account_data = row_data["account_row_id"]
        customer_data = row_data["customer"]

        print(account_data)
        for a_customer in customer_data:
            # print (a_customer.keys())

            # print(account_data)

            # print(a_customer["customer_id"])
            # print(a_customer["email"])

            transactions_data = a_customer["transactions"]
            # print(transactions_data.keys())

            write_transactions(a_customer["customer_id"],transactions_data)


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

            print(" ** ")
            # print(transactions_data[0].keys())
            # for a_transaction in transactions_data:
            #     print(a_transaction["amount"])

        account_balance_data = row_data["account_balance"]

        account_spend_limit_data = row_data["account_spend_limit"]

        rewards_data = row_data["rewards"]

        total_reward_remaining_data = row_data["total_rewards_remaining"]

        payments_data = row_data["payments"]

        is_original_data = row_data["is_original"]

        print ("##")
        # if index == 1 :
        #     break

        # index = index +1

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
