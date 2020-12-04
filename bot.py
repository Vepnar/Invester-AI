import time
from ai import scaler 
import ai.data_handler as dh
from datetime import datetime
from ai.predict import load_model
import ai.financial_api as financial

# Illegal inport
from settings import *

BETS_REMAINING = 1
BETS_IN = 0
LAST_BET = 0
BALANCE = 100
BALANCE_BETS = 0
sleep = 30*60
LATEST_ACTION = 0

ACTION_DICT = {
    0 : 'wait',
    1 : 'pull',
    2 : 'invest'
}



def compute_action(today, tomorrow):
    global LATEST_ACTION, LAST_BET, BETS_REMAINING, BETS_IN

    computed_gain = ((sum(tomorrow) / 2) - today[1]) / (sum(tomorrow) / 2)

    # 0 = wait
    # 1 = pull
    # 2 = invest
    raw_action, rule_based_action = 0, 0

    # Calculate action based on gain
    if abs(computed_gain) > MIN_G_CHANGE :
        
        if computed_gain > 0:
            raw_action = 2
        else:
            raw_action = 1

    if (raw_action == 1 and today[1] > LAST_BET and BETS_IN > 0):
        rule_based_action = 1

    elif (raw_action == 2 and BETS_REMAINING > 0 and LATEST_ACTION is not 2):
        rule_based_action = 2

    return rule_based_action, computed_gain * 100

def main():
    global LAST_BET, BETS_REMAINING, LATEST_ACTION, BALANCE_BETS, BALANCE, BETS_IN
    print('Loading neural network and settings...')
    model = load_model()

    counter = 1
    logger = open('logs.csv', 'a')
    print('Running neural network')
    while 1:
        # Check if the dataset is too old
        if(dh.dataset_age() > 0):
            print(f'Dataset is {dh.dataset_age()} days old and can\'t be used')
            break


        print('============================================')
        window = dh.latest_window()
        

        # Recieve the values from yesterday
        yesterday = scaler.unscale_x([window[0][1]])[0][:2]

        # Recieve values for today
        #today = dh.unscale_x([window[0][0]])[0][:2]
        try:
            today = financial.recieve_updates(data=UPDATE_WINDOW)
        except Exception as e:
            print('Recieving information failed: ', e)
            continue
        
        window[0] = scaler.scale_x([today])
        today = today[:2]
        


        # Predict a new future
        tomorrow = scaler.unscale_y(model.predict(window))[0]

        computed_action, gain = compute_action(today, tomorrow)

        date = datetime.now().strftime('%d %h %H:%M')
        print(f'Time: {date}, Iterations: {counter}')
        print(f'Today: {today}, yesterday: {yesterday}, tomorrow: {tomorrow}')
        print(f'Action: {ACTION_DICT[computed_action]} GP: {gain}')


        

        if computed_action == 2:
            LAST_BET = today[1]
            BETS_REMAINING-= 1
            BETS_IN +=1
            BALANCE_BETS = 100
            BALANCE -= 100
        elif computed_action == 1:
            BETS_REMAINING+= 1
            BETS_IN -=1
            BALANCE = BALANCE_BETS / LAST_BET * today[1]
            LAST_BET = 0
            BALANCE_BETS = 0

        bets_value = 0
        if(BALANCE_BETS > 0 and LAST_BET > 0):
            bets_value = BALANCE_BETS / LAST_BET * today[1]
            
        print(f'Balance: ${BALANCE}, Bets: ${bets_value}, Total: ${BALANCE + bets_value}')
        logger.write(f'{date},{BALANCE},{LAST_BET},{BETS_IN},{bets_value},{ACTION_DICT[computed_action]},{today[0]},{today[1]},{tomorrow[0]},{tomorrow[1]}\n')
        logger.flush()
        
        LATEST_ACTION = compute_action
        
        counter+=1
        time.sleep(sleep)
        

if __name__ == '__main__':
    main()
    