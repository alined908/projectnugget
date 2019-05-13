import os
import pandas as pd
import ast

def get_map_info(csv_path):
    df = pd.read_csv(csv_path, sep = ",")
    date = df.loc[0]['Date']
    map = df.loc[0]['Map']
    opponent = df.loc[0]['Opponent']
    total_game_time = df.loc[len(df) - 1]['Duration']

    return date, map, opponent, total_game_time

def get_ttcu_ttuu(csv_path, team1, team2):
    df = pd.read_csv(csv_path, sep = ",")
    tracker = []
    final_ttcu = {'Player 1': {}, 'Player 2': {}, 'Player 3': {}, 'Player 4': {}, 'Player 5': {}, 'Player 6': {},
    'Player 7': {}, 'Player 8': {}, 'Player 9': {}, 'Player 10': {}, 'Player 11': {}, 'Player 12': {}}
    final_ttuu = {'Player 1': {}, 'Player 2': {}, 'Player 3': {}, 'Player 4': {}, 'Player 5': {}, 'Player 6': {},
    'Player 7': {}, 'Player 8': {}, 'Player 9': {}, 'Player 10': {}, 'Player 11': {}, 'Player 12': {}}

    #Iterate over dataframe rows
    for count, (i, row) in enumerate(df.iterrows()):
        curr_row_time = row['Duration']
        curr_gamestate = row['GameState']

        #Do something if last row
        if count == (len(df) - 1):
            pass
        else:
            next_row_time = df.loc[i+1]['Duration']

        #Skip if have unknownhero or if in lobby
        if curr_gamestate == 'lobby' or (curr_row_time == 0 and next_row_time == 0):
            continue

        prev_row_time = df.loc[i-1]['Duration']
        #Check which ult_charge_state changed
        for index in range(1, 13):
            #Might need these variables
            prev_hero = df.loc[i-1]['Hero ' + str(index)]
            curr_hero = row['Hero ' + str(index)]
            prev_charge = df.loc[i-1]['Ult_Charge ' + str(index)]
            curr_charge = row['Ult_Charge ' + str(index)]

            #Game starts, put current status of [hero, uncharged, 0] into tracker
            if curr_row_time == 0 and next_row_time > 0:
                player = 'Player ' + str(index)
                hero = 'Hero ' + str(index)
                tracker.append([curr_hero, curr_charge, 0])

            else:
                #Check if a hero switch happened and change tracker accordingly
                if curr_hero != prev_hero:
                    tracker[index-1] = [curr_hero, 'uncharged', curr_row_time]

                #If no hero switch happened
                else:
                    #Check if ult is charged up and if so add to ttcu
                    if curr_charge == 'charged' and prev_charge == 'uncharged':
                        ttcu = curr_row_time - tracker[index - 1][2]
                        if ttcu < 10:
                            continue
                        if curr_hero in final_ttcu['Player ' + str(index)].keys():
                            final_ttcu['Player ' + str(index)][curr_hero].append(ttcu)
                        else:
                            final_ttcu['Player ' + str(index)][curr_hero] = [ttcu]
                        tracker[index-1][1] = 'charged'
                        tracker[index-1][2] = curr_row_time

                    #Check if ult is used and if so add to ttuu
                    if curr_charge == 'uncharged' and prev_charge == 'charged':
                        ttuu = curr_row_time - tracker[index - 1][2]
                        if curr_hero in final_ttuu['Player ' + str(index)].keys():
                            final_ttuu['Player ' + str(index)][curr_hero].append(ttuu)
                        else:
                            final_ttuu['Player ' + str(index)][curr_hero] = [ttuu]
                        tracker[index-1][1] = 'uncharged'
                        tracker[index-1][2] = curr_row_time

    #Split by teams
    final_team1_ttcu, final_team2_ttcu = dict(list(final_ttcu.items())[:len(final_ttcu)//2]), dict(list(final_ttcu.items())[len(final_ttcu)//2:])
    final_team1_ttuu, final_team2_ttuu = dict(list(final_ttuu.items())[:len(final_ttuu)//2]), dict(list(final_ttuu.items())[len(final_ttuu)//2:])

    for num, name in enumerate(team1):
        final_team1_ttcu[name] = final_team1_ttcu.pop('Player ' + str(num + 1))
        final_team1_ttuu[name] = final_team1_ttuu.pop('Player ' + str(num + 1))

    for num, name in enumerate(team2):
        final_team2_ttcu[name] = final_team2_ttcu.pop('Player ' + str(num + 7))
        final_team2_ttuu[name] = final_team2_ttuu.pop('Player ' + str(num + 7))

    return final_team1_ttcu, final_team2_ttcu, final_team1_ttuu, final_team2_ttuu

def get_teamcomps(csv_path):
    df = pd.read_csv(csv_path, sep = ",")
    team1_comp, team2_comp = [], []
    team1_duration, team2_duration = [], []
    final_team1_comps, final_team2_comps = {}, {}
    prev_team1_time, prev_team2_time = 0, 0
    first_flag = True

    #Iterate over dataframe rows
    for count, (i, row) in enumerate(df.iterrows()):
        curr_row_time = row['Duration']
        curr_gamestate = row['GameState']
        team1_hero_change = []
        team2_hero_change = []
        skip_flag = False

        #Add comps,time  from last row
        if count == (len(df) - 1):
            team1_comp.append([hero1, hero2, hero3, hero4, hero5, hero6])
            team1_duration.append(curr_row_time - prev_team1_time)
            team2_comp.append([hero7, hero8, hero9, hero10, hero11, hero12])
            team2_duration.append(curr_row_time - prev_team2_time)

        #Skip if have unknownhero or if in lobby
        if curr_gamestate == 'lobby' or curr_row_time == 0:
            continue

        #Check if hero swap
        for index in range(1,13):
            prev_hero = df.loc[i-1]['Hero ' + str(index)]
            curr_hero = row['Hero ' + str(index)]
            if curr_hero != prev_hero and index < 7:
                team1_hero_change.append(index)
            elif curr_hero != prev_hero and index >= 7:
                team2_hero_change.append(index)

        #Last row elements
        hero1, hero2, hero3 = df.loc[i-1]['Hero 1'], df.loc[i-1]['Hero 2'], df.loc[i-1]['Hero 3']
        hero4, hero5, hero6 = df.loc[i-1]['Hero 4'], df.loc[i-1]['Hero 5'], df.loc[i-1]['Hero 6']
        hero7, hero8, hero9 = df.loc[i-1]['Hero 7'], df.loc[i-1]['Hero 8'], df.loc[i-1]['Hero 9']
        hero10, hero11, hero12 = df.loc[i-1]['Hero 10'], df.loc[i-1]['Hero 11'], df.loc[i-1]['Hero 12']

        #If no hero_swap go to next row, otherwise save last comp and time played
        if len(team1_hero_change) == 0 and len(team2_hero_change) == 0:
            continue
        if len(team1_hero_change) > 0:
            team1_comp.append([hero1, hero2, hero3, hero4, hero5, hero6])
            team1_duration.append(curr_row_time - prev_team1_time)
            prev_team1_time = df.loc[i]['Duration']
        if len(team2_hero_change) > 0:
            team2_comp.append([hero7, hero8, hero9, hero10, hero11, hero12])
            team2_duration.append(curr_row_time - prev_team2_time)
            prev_team2_time = df.loc[i]['Duration']

    #Create Final Set
    for num, comp in enumerate(team1_comp):
        time_played = team1_duration[num]
        if time_played < 5 or ('unknownhero' in comp):
            continue
        else:
            if tuple(comp) in final_team1_comps:
                final_team1_comps[tuple(comp)] += time_played
            else:
                final_team1_comps[tuple(comp)] = time_played

    for num2, comp2 in enumerate(team2_comp):
        time_played = team2_duration[num2]
        if time_played < 5 or ('unknownhero' in comp2):
            continue
        else:
            if tuple(comp2) in final_team2_comps:
                final_team2_comps[tuple(comp2)] += time_played
            else:
                final_team2_comps[tuple(comp2)] = time_played

    return final_team1_comps, final_team2_comps

def get_rosters(csv_path):
    df = pd.read_csv(csv_path, sep = ",")

    left_team, right_team = [], []
    for num in range(1,13):
        if num < 7:
            left_team.append(df.loc[len(df)//2]['Name ' + str(num)])
        else:
            right_team.append(df.loc[len(df)//2]['Name ' + str(num)])

    return left_team, right_team

def get_kill_deaths(csv_path, team1, team2):
    df = pd.read_csv(csv_path, sep = ",")
    kd_dict = {key: {'Total': {'Kills': 0, 'Deaths': 0}} for key in (team1+team2)}

    for count, (i, row) in enumerate(df.iterrows()):
        kill_array, death_array= ast.literal_eval(row['Kills']), ast.literal_eval(row['Deaths'])
        kill_name, kill_hero, death_name, death_hero = kill_array[0], kill_array[1], death_array[0], death_array[1]

        if kill_name == "" and death_name == "":
            continue

        #Suicide case
        if not kill_hero == "":
            if kill_hero in kd_dict[kill_name]:
                kd_dict[kill_name][kill_hero]['Kills'] += 1
            else:
                kd_dict[kill_name][kill_hero] = {'Kills': 1, 'Deaths': 0}
            kd_dict[kill_name]['Total']['Kills'] += 1

        if death_hero in kd_dict[death_name]:
            kd_dict[death_name][death_hero]['Deaths'] += 1
        else:
            kd_dict[death_name][death_hero] = {'Kills': 0, 'Deaths': 1}
        kd_dict[death_name]['Total']['Deaths'] += 1

    team1_kd, team2_kd = dict(list(kd_dict.items())[:len(kd_dict)//2]), dict(list(kd_dict.items())[len(kd_dict)//2:])
    return team1_kd, team2_kd

def get_fight_stats(csv_path, team1, team2):
    #To be implemented
    return

if __name__ == '__main__':
    csv_folder = "csvs/to_csv/"
    for csv in os.listdir(csv_folder):
        csv_path = csv_folder + csv
        date, map, opponent, total_game_time = get_map_info(csv_path)
        team1, team2 = get_rosters(csv_path)
        team1_comps, team2_comps = get_teamcomps(csv_path)
        team1_ttcu, team2_ttcu, team1_ttuu, team2_ttuu = get_ttcu_ttuu(csv_path, team1, team2)
        team1_kd, team2_kd = get_kill_deaths(csv_path, team1, team2)
        print("===============================================")
        print("Match Summary")
        print("===============================================")
        print("Date:", date)
        print("Map:", map)
        print("Opponent:", opponent)
        print("Duration:", total_game_time)
        print("Team 1:", " ".join(team1))
        print("Team 2:", " ".join(team2))
        print('===============================================')
        print("Team 1 Comps: ")
        print(team1_comps)
        print('------------------------------------------------------------------------------------------------')
        print("Team 2 Comps: ")
        print(team2_comps)
        print('------------------------------------------------------------------------------------------------')
        print("Team 1 Kills/Deaths: ")
        print(team1_kd)
        print('------------------------------------------------------------------------------------------------')
        print("Team 2 Kills/Deaths: ")
        print(team2_kd)
        print('------------------------------------------------------------------------------------------------')
        print("Team 1 TTCU: ")
        print(team1_ttcu)
        print('------------------------------------------------------------------------------------------------')
        print("Team 2 TTCU: ")
        print(team2_ttcu)
        print('------------------------------------------------------------------------------------------------')
        print("Team 1 TTUU: ")
        print(team1_ttuu)
        print('------------------------------------------------------------------------------------------------')
        print("Team 2 TTUU: ")
        print(team2_ttuu)
        print('------------------------------------------------------------------------------------------------')
