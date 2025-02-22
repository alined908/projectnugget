import os
import pandas as pd
import ast

map_dic = {'BUSAN': 'koth', 'ILIOS': 'koth', 'LJ': 'koth', 'NEPAL': 'koth', 'OASIS': 'koth',
'NUMBANI': 'non-koth', 'KR': 'non-koth', 'BW': 'non-koth', 'KR': 'non-koth', 'HOLLYWOOD': 'non-koth', 'HOLLY': 'non-koth',
'R66': 'non-koth', 'GIB': 'non-koth', 'RIALTO': 'non-koth', 'DORADO': 'non-koth', 'JUNKERTOWN': 'non-koth',
'HANAMURA': 'non-koth', 'LUNAR': 'non-koth', 'PARIS': 'non-koth', 'VOLSKAYA': 'non-koth', 'ANUBIS': 'non-koth'}
shock_roster = ["SUPER", "CHOIHYOBIN", "SINATRAA", "VIOL2T", "MOTH", "RASCAL", "ARCHITECT", "STRIKER", "SMURF", "NEVIX"]

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
        if count != (len(df) - 1):
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

def get_teamcomps(csv_path, map, team1, team2):
    df = pd.read_csv(csv_path, sep = ",")
    team1_comp, team2_comp = [], []
    team1_duration, team2_duration = [], []
    final_team1_comps, final_team2_comps = {}, {}
    prev_team1_time, prev_team2_time = 0, 0
    first_flag, shock_is_team1_flag = True, False

    #Check which team is shock
    for shock in shock_roster:
        if shock in team1:
            shock_is_team1_flag = True
            break

    #Iterate over dataframe rows
    for count, (i, row) in enumerate(df.iterrows()):
        curr_row_time, curr_gamestate = row['Duration'], row['GameState']
        team1_hero_change, team2_hero_change  = [], []
        skip_flag = False

        #Add comps,time  from last row
        if count == (len(df) - 1):
            if map_dic[map] == 'koth':
                team1_array = [roundtype, hero1, hero2, hero3, hero4, hero5, hero6]
                team2_array = [roundtype, hero7, hero8, hero9, hero10, hero11, hero12]
            else:
                if roundtype == "ATTACK":
                    opponent_roundtype = "DEFENSE"
                if roundtype == "DEFENSE":
                    opponent_roundtype = "ATTACK"
                if shock_is_team1_flag:
                    team1_array = [roundtype, hero1, hero2, hero3, hero4, hero5, hero6]
                    team2_array = [opponent_roundtype, hero7, hero8, hero9, hero10, hero11, hero12]
                else:
                    team1_array = [opponent_roundtype, hero1, hero2, hero3, hero4, hero5, hero6]
                    team2_array = [roundtype, hero7, hero8, hero9, hero10, hero11, hero12]
            team1_comp.append(team1_array)
            team1_duration.append(curr_row_time - prev_team1_time)
            team2_comp.append(team2_array)
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
        roundtype = df.loc[i-1]['Roundtype']


        #If no hero_swap go to next row, otherwise save last comp and time played
        if len(team1_hero_change) == 0 and len(team2_hero_change) == 0:
            continue

        if map_dic[map] == 'koth':
            team1_array = [roundtype, hero1, hero2, hero3, hero4, hero5, hero6]
            team2_array = [roundtype, hero7, hero8, hero9, hero10, hero11, hero12]
        else:
            if roundtype == "ATTACK":
                opponent_roundtype = "DEFENSE"
            if roundtype == "DEFENSE":
                opponent_roundtype = "ATTACK"
            if shock_is_team1_flag:
                team1_array = [roundtype, hero1, hero2, hero3, hero4, hero5, hero6]
                team2_array = [opponent_roundtype, hero7, hero8, hero9, hero10, hero11, hero12]
            else:
                team1_array = [opponent_roundtype, hero1, hero2, hero3, hero4, hero5, hero6]
                team2_array = [roundtype, hero7, hero8, hero9, hero10, hero11, hero12]

        if len(team1_hero_change) > 0:
            team1_comp.append(team1_array)
            team1_duration.append(curr_row_time - prev_team1_time)
            prev_team1_time = df.loc[i]['Duration']
        if len(team2_hero_change) > 0:
            team2_comp.append(team2_array)
            team2_duration.append(curr_row_time - prev_team2_time)
            prev_team2_time = df.loc[i]['Duration']

    #Create Final Set
    for num, comp in enumerate(team1_comp):
        time_played = team1_duration[num]
        roundtype, short_comp = comp[0], comp[1:]
        if time_played < 5 or ('unknownhero' in comp):
            continue
        else:
            if roundtype not in final_team1_comps:
                final_team1_comps[roundtype] = {}

            if tuple(short_comp) in final_team1_comps[roundtype]:
                final_team1_comps[roundtype][tuple(short_comp)] += time_played
            else:
                final_team1_comps[roundtype][tuple(short_comp)] = time_played

    for num, comp in enumerate(team2_comp):
        time_played = team2_duration[num]
        roundtype, short_comp = comp[0], comp[1:]
        if time_played < 5 or ('unknownhero' in comp):
            continue
        else:
            if roundtype not in final_team2_comps:
                final_team2_comps[roundtype] = {}
            if tuple(short_comp) in final_team2_comps[roundtype]:
                final_team2_comps[roundtype][tuple(short_comp)] += time_played
            else:
                final_team2_comps[roundtype][tuple(short_comp)] = time_played

    comps1_dict = {'Team': [], 'Roundname': [], 'Composition': [], 'Duration': []}
    comps2_dict = {'Team': [], 'Roundname': [], 'Composition': [], 'Duration': []}
    for roundtype in final_team1_comps:
        for comp in final_team1_comps[roundtype]:
            comps1_dict['Roundname'].append(roundtype)
            comps1_dict['Composition'].append(comp)
            comps1_dict['Duration'].append(final_team1_comps[roundtype][comp])
            comps1_dict['Team'].append(team1)

    for roundtype in final_team2_comps:
        for comp in final_team2_comps[roundtype]:
            comps2_dict['Roundname'].append(roundtype)
            comps2_dict['Composition'].append(comp)
            comps2_dict['Duration'].append(final_team2_comps[roundtype][comp])
            comps2_dict['Team'].append(team2)

    return comps1_dict, comps2_dict

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

def append_fight_stats(general_fight_dict, kill_array, death_array, curr_row_time, timestamp, team1, team2, team1_hero_roster, team2_hero_roster,
total_kill_sequence, team1_kill_sequence, team2_kill_sequence, total_death_sequence, team1_death_sequence, team2_death_sequence,
total_ult_sequence, team1_ult_sequence, team2_ult_sequence, map, roundtype, fight_player_dict):

    kill_name, kill_hero, death_name, death_hero = kill_array[0], kill_array[1], death_array[0], death_array[1]

    #First Blood
    roster = []
    first_blood_name = total_kill_sequence[0][0]
    if first_blood_name in team1:
        roster = team1
    else:
        roster = team2

    #Winner
    winners, losers, winners_kill_sequence, winners_death_sequence, losers_kill_sequence, losers_death_sequence = [], [], [], [], [], []
    if len(team1_kill_sequence) > len(team2_kill_sequence):
        winners, losers, winner_heroes, loser_heroes = team1, team2, team1_hero_roster, team2_hero_roster
        winners_kill_sequence, losers_kill_sequence, winners_death_sequence, losers_death_sequence = team1_kill_sequence, team2_kill_sequence, team1_death_sequence, team2_death_sequence
    elif len(team1_kill_sequence) < len(team2_kill_sequence):
        winners, losers, winner_heroes, loser_heroes = team2, team1, team2_hero_roster, team1_hero_roster
        winners_kill_sequence, losers_kill_sequence, winners_death_sequence, losers_death_sequence = team2_kill_sequence, team1_kill_sequence, team2_death_sequence, team1_death_sequence
    else:
        pass
        #print("SAME # OF KILLS")

    #Fight general information
    general_fight_dict['Map'].append(map)
    general_fight_dict['Roundtype'].append(roundtype)
    general_fight_dict['Length'].append(curr_row_time - timestamp)
    general_fight_dict['Winner'].append(winners)
    general_fight_dict['L Players'].append(team1)
    general_fight_dict['R Players'].append(team2)
    general_fight_dict['L Heroes'].append(team1_hero_roster)
    general_fight_dict['R Heroes'].append(team2_hero_roster)
    #Fight K/D Information
    general_fight_dict['Total Kill Sequence'].append(total_kill_sequence)
    general_fight_dict['L Kill Sequence'].append(team1_kill_sequence)
    general_fight_dict['R Kill Sequence'].append(team2_kill_sequence)
    general_fight_dict['Total Death Sequence'].append(total_death_sequence)
    general_fight_dict['L Death Sequence'].append(team1_death_sequence)
    general_fight_dict['R Death Sequence'].append(team2_death_sequence)
    general_fight_dict['L Kill #'].append(len(team1_kill_sequence))
    general_fight_dict['R Kill #'].append(len(team2_kill_sequence))
    general_fight_dict['First Blood'].append(roster)
    #Fight Ult Information
    general_fight_dict['L Ult Sequence'].append(team1_ult_sequence)
    general_fight_dict['R Ult Sequence'].append(team2_ult_sequence)
    general_fight_dict['L # Ults Used'].append(len(team1_ult_sequence))
    general_fight_dict['R # Ults Used'].append(len(team2_ult_sequence))
    #general_fight_dict['L # Ults Avail'].append()
    #general_fight_dict['R # Ults Avail'].append()

    try:
        winners_first_kill_name = winners_kill_sequence[0][0]
    except IndexError:
        winners_first_kill_name = ""
    try:
        winners_first_death_name = winners_death_sequence[0][0]
    except IndexError:
        winners_first_death_name = ""
    try:
        losers_first_kill_name = losers_kill_sequence[0][0]
    except IndexError:
        losers_first_kill_name = ""
    try:
        losers_first_death_name = losers_death_sequence[0][0]
    except IndexError:
        losers_first_death_name = ""

    #Fight player Information
    for index, player in enumerate(winners):
        hero = winner_heroes[index]

        if hero in fight_player_dict[player]['Heroes']:
            fight_player_dict[player]['Heroes'][hero]['Win'] += 1
            fight_player_dict[player]['Heroes'][hero]['Total'] += 1
        else:
            fight_player_dict[player]['Heroes'][hero] = {'Win': 1, 'Lose': 0, 'Total': 1, 'First Kill': 0, 'First Death' : 0}

        if player == winners_first_kill_name:
            fight_player_dict[player]['Heroes'][hero]['First Kill'] += 1
            fight_player_dict[player]['Totals']['First Kill'] += 1
        if player == winners_first_death_name:
            fight_player_dict[player]['Heroes'][hero]['First Death'] += 1
            fight_player_dict[player]['Totals']['First Death'] += 1

        fight_player_dict[player]['Totals']['Win'] += 1
        fight_player_dict[player]['Totals']['Total'] += 1

    for index, player in enumerate(losers):
        hero = loser_heroes[index]

        if hero in fight_player_dict[player]['Heroes']:
            fight_player_dict[player]['Heroes'][hero]['Lose'] += 1
            fight_player_dict[player]['Heroes'][hero]['Total'] += 1
        else:
            fight_player_dict[player]['Heroes'][hero] = {'Win': 0, 'Lose': 1, 'Total': 1, 'First Kill': 0, 'First Death' : 0}

        if player == losers_first_kill_name:
            fight_player_dict[player]['Heroes'][hero]['First Kill'] += 1
            fight_player_dict[player]['Totals']['First Kill'] += 1
        if player == losers_first_death_name:
            fight_player_dict[player]['Heroes'][hero]['First Death'] += 1
            fight_player_dict[player]['Totals']['First Death'] += 1

        fight_player_dict[player]['Totals']['Lose'] += 1
        fight_player_dict[player]['Totals']['Total'] += 1

    # print("Map:", map, "--> Roundtype (SFS):", roundtype)
    # print("Length of Fight: " + str(curr_row_time - timestamp), "====> First Blood: ", total_kill_sequence[0], "====> Winner: ", winners)
    # print("Team 1 Hero Roster:", team1_hero_roster)
    # print("Team 2 Hero Roster:", team2_hero_roster)
    # print("All Kills:" , total_kill_sequence)
    # print("All Deaths:" , total_death_sequence)
    # print("Team 1 Kill:" , team1_kill_sequence, "====>", "Team 1 Kill #:", len(team1_kill_sequence))
    # print("Team 1 Death:" , team1_death_sequence)
    # print("Team 2 Kill:" , team2_kill_sequence, "====>", "Team 2 Kill #:", len(team2_kill_sequence))
    # print("Team 2 Death:" , team2_death_sequence)
    # print("Total Ult Sequence:", total_ult_sequence)
    # print("Team 1 Ult Sequence:", team1_ult_sequence)
    # print("Team 2 Ult Sequence:", team2_ult_sequence)
    # print("Fight End (" + str(curr_row_time)+"): "  + kill_name + ": " + kill_hero + " --> " + death_name + ": " + death_hero)

    #print("Team 1 First Kill:", team1_kill_sequence[0][0], " --> ", "Team 1 First Death:" ,team1_death_sequence[0][0])
    #print("Team 2 First Death:", team1_kill_sequence[0][0], " --> ", "Team 1 First Death:", team1_death_sequence[0][0])
    # print("==================================================================================================")

#1. General Fight Information: Map, (If need be manually added roundtype), Length of Fight, Left/Right Hero Roster, Left/Right Player Roster, Left Team/Right Team Kill/Death #,
# Left/Right # of Ults Available, Left Team/Right Team # of Ults Used (And which ones), First Blood (which team?), Ult Sequence, Death Sequence
#2. Player Specific: First Kill %, First Death %
#3  Hero Specific: Fight Win % When Hero Dies First
#4. From #1 -- > Team(Roster of 6) First Blood, First Blood Win %, First Death, First Death Win %
#5. From #1 --> Fight Win % when X ults more/less used
# NOTE: Be cautious of it currently counting the next row after fight ends (last kill) for ult. Differs from winston's lab implementation, but makes sense i think..Probably should add a time check
# Note: Be cautious of how dva ult is tracked
def get_fight_stats(csv_path, team1, team2, map):
    df = pd.read_csv(csv_path, sep = ",")
    fight_player_dict = {key: {'Totals': {'Win': 0, 'Lose': 0, 'Total': 0, 'First Kill': 0, 'First Death' : 0}, 'Heroes': {}} for key in (team1+team2)}
    general_fight_dict = {'Map': [], 'Roundtype': [], 'Length': [], 'L Players': [], 'L Heroes': [], 'R Players': [], 'R Heroes': [], 'First Blood': [], 'Winner': [],
    'L Kill #': [], 'R Kill #': [], 'L # Ults Used': [], 'R # Ults Used': [], 'Total Kill Sequence': [], 'L Kill Sequence': [], 'R Kill Sequence': [],
    'Total Death Sequence': [], 'L Death Sequence': [], 'R Death Sequence': [], 'L Ult Sequence': [], 'R Ult Sequence': []}
    in_fight = False

    for count, (i, row) in enumerate(df.iterrows()):
        kill_array, death_array= ast.literal_eval(row['Kills']), ast.literal_eval(row['Deaths'])
        kill_name, kill_hero, death_name, death_hero = kill_array[0], kill_array[1], death_array[0], death_array[1]
        curr_row_time, curr_gamestate = row['Duration'], row['GameState']

        #Skip if have unknownhero or if in lobby
        if curr_gamestate == 'lobby' or (curr_row_time == 0 and df.loc[i+1]['Duration'] == 0):
            continue

        if not in_fight:
            if death_name == "" and death_hero == "":
                continue
            else:
                first_death_timestamp, length_tracker, countdown = row['Duration'], 0, 13
                in_fight, first_ult_timestamp, roundtype = True, float('inf'), row['Roundtype']
                total_kill_sequence, total_death_sequence, total_ult_sequence = [], [], []
                team1_kill_sequence, team1_death_sequence, team1_ult_sequence = [], [], []
                team2_kill_sequence, team2_death_sequence, team2_ult_sequence = [], [], []

                #Append hero roster
                team1_hero_roster, team2_hero_roster = [], []
                for index in range(1, 13):
                    curr_hero = row['Hero ' + str(index)]
                    team1_hero_roster.append(curr_hero) if (index < 7) else team2_hero_roster.append(curr_hero)

                #Previous rows ult check (do on row of fight start)
                row_num = i
                ult_refresh_tracker = ['ready'] * 12
                while True:
                    prev_row_time, ultimates_used_index = df.loc[row_num-1]['Duration'], []
                    prev_row_time_difference = curr_row_time - prev_row_time

                    if prev_row_time_difference > 12:
                        break

                    for index in range(1, 13):
                        prev_hero, prev_charge = df.loc[row_num-1]['Hero ' + str(index)], df.loc[row_num-1]['Ult_Charge ' + str(index)]
                        curr_hero, curr_charge = row['Hero ' + str(index)], row['Ult_Charge ' + str(index)]
                        name, ult_status = row['Name ' + str(index)], ult_refresh_tracker[index - 1]
                        if curr_hero == prev_hero:
                            if curr_charge == 'uncharged' and prev_charge == 'charged' and ult_status == 'ready':
                                first_ult_timestamp = df.loc[row_num]['Duration']
                                total_ult_sequence.insert(0, [name, prev_hero, first_ult_timestamp])
                                ultimates_used_index.insert(0, index)
                                ult_refresh_tracker[index - 1] = 'not ready'
                            if curr_charge == 'charged' and prev_charge == 'uncharged' and ult_status == 'not ready':
                                ult_refresh_tracker[index - 1] = 'ready'

                    for num in ultimates_used_index:
                        team1_ult_sequence.insert(0, [row['Name ' + str(num)], row['Hero ' + str(num)]]) if (num < 7) else team2_ult_sequence.insert(0, [row['Name ' + str(num)], row['Hero ' + str(num)]])

                    row_num -= 1

                timestamp = min(first_death_timestamp, first_ult_timestamp)
                #print("Ult Usage Before Fight Start: ", total_ult_sequence)
                #print("Fight Start ("+ str(timestamp)+"): "  + kill_name + ": " + kill_hero + " --> " + death_name + ": " + death_hero) if (first_death_timestamp <= first_ult_timestamp) else print("Fight Start ("+ str(timestamp)+"):" , "Ult Used --> " + total_ult_sequence[0][0] + ": "+ total_ult_sequence[0][1])

        if in_fight:
            prev_row_time = df.loc[i-1]['Duration']
            prev_row_time_difference = curr_row_time - prev_row_time
            countdown -= prev_row_time_difference

            #If another kill, reset countdown and add to appropriate kill/death arrays
            if kill_name != "" and kill_hero != "":
                countdown = 13
                total_kill_sequence.append(kill_array)
                total_death_sequence.append(death_array)
                if kill_name in team1:
                    team1_kill_sequence.append(kill_array)
                    team2_death_sequence.append(death_array)
                else:
                    team2_kill_sequence.append(kill_array)
                    team1_death_sequence.append(death_array)

            #Current row ult check
            for index in range(1, 13):
                next_hero, next_charge = df.loc[i+1]['Hero ' + str(index)], df.loc[i+1]['Ult_Charge ' + str(index)]
                curr_hero, curr_charge = row['Hero ' + str(index)], row['Ult_Charge ' + str(index)]
                name, ult_status = row['Name ' + str(index)], ult_refresh_tracker[index - 1]

                if curr_hero == next_hero:
                    if curr_charge == 'charged' and next_charge == 'uncharged' and ult_status == 'ready':
                        ult_refresh_tracker[index - 1] = 'not ready'
                        #print([name, next_hero, df.loc[i+1]['Duration']])
                        total_ult_sequence.append([name, next_hero, df.loc[i+1]['Duration']])
                        ultimates_used_index.append(index)
                    if curr_charge == 'uncharged' and next_charge == 'charged' and ult_status == 'not ready':
                        ult_refresh_tracker[index - 1] = 'ready'


            #Check future rows and see when fight ends aka last kill is made
            row_num = i
            while countdown > 0:
                next_row_time, next_kill_array = df.loc[row_num+1]['Duration'], ast.literal_eval(df.loc[row_num+1]['Kills'])
                next_kill_name, next_kill_hero = next_kill_array[0], next_kill_array[1]
                next_row_time_difference = next_row_time - curr_row_time

                if (next_row_time_difference <= 13) and (next_kill_name != "" and next_kill_hero != ""):
                    break
                if (next_row_time_difference > 13) or (df.loc[row_num+1]['GameState'] == 'lobby'):
                    countdown = 0
                row_num += 1

            #Fight is over
            if countdown <= 0:
                in_fight = False

                #Ult Usage Sequence
                for num in ultimates_used_index:
                    team1_ult_sequence.append([row['Name ' + str(num)], row['Hero ' + str(num)]]) if (num < 7) else team2_ult_sequence.append([row['Name ' + str(num)], row['Hero ' + str(num)]])

                #Append fight info to general fight dict
                append_fight_stats(general_fight_dict, kill_array, death_array, curr_row_time, timestamp, team1, team2, team1_hero_roster, team2_hero_roster,
                total_kill_sequence, team1_kill_sequence, team2_kill_sequence, total_death_sequence, team1_death_sequence, team2_death_sequence, total_ult_sequence, team1_ult_sequence, team2_ult_sequence, map, roundtype, fight_player_dict)

    return general_fight_dict, fight_player_dict

def print_example(csv_path, date, map, opponent, total_game_time, team1, team2, team1_comps, team2_comps, team1_ttcu, team2_ttcu, team1_kd, team2_kd, general_fight_stats):
    print("===============================================")
    print("Match Summary: " + csv_path)
    print("===============================================")
    print("Date:", date)
    print("Map:", map)
    print("Opponent:", opponent)
    print("Duration:", total_game_time)
    print("Team 1:", " ".join(team1))
    print("Team 2:", " ".join(team2))
    #print("===============================================")
    # print("Match Basic Statistics")
    # print("===============================================")
    # print("Team 1 Comps: ")
    # print(team1_comps)
    # print('------------------------------------------------------------------------------------------------')
    # print("Team 2 Comps: ")
    # print(team2_comps)
    # print('------------------------------------------------------------------------------------------------')
    # print("Team 1 Kills/Deaths: ")
    # print(team1_kd)
    # print('------------------------------------------------------------------------------------------------')
    # print("Team 2 Kills/Deaths: ")
    # print(team2_kd)
    # print('------------------------------------------------------------------------------------------------')
    # print("Team 1 TTCU: ")
    # print(team1_ttcu)
    # print('------------------------------------------------------------------------------------------------')
    # print("Team 2 TTCU: ")
    # print(team2_ttcu)
    # print('------------------------------------------------------------------------------------------------')
    # print("Team 1 TTUU: ")
    # print(team1_ttuu)
    # print('------------------------------------------------------------------------------------------------')
    # print("Team 2 TTUU: ")
    # print(team2_ttuu)
    # print('------------------------------------------------------------------------------------------------')
    # print("===============================================")
    # print("Match Fight Statistics")
    # print("===============================================")
    # print(general_fight_stats)

def create_general_dict(team1_ttcu, team2_ttcu, team1_ttuu, team2_ttuu, team1_kd, team2_kd, player_fight_stats):
    general_dict = {'Name': [], 'Hero': [], 'Kill': [], 'Death': [], 'TTCU': [], 'TTUU': [], 'Fight #': [], 'Fight Win': [], 'Fight Lose': [], 'First Kill': [], 'First Death': []}
    ttcu_dict, ttuu_dict, kd_dict = dict(team1_ttcu, **team2_ttcu), dict(team1_ttuu, **team2_ttuu), dict(team1_kd, **team2_kd)

    for name in kd_dict:
        for hero in kd_dict[name]:
            if hero == 'Total':
                continue
            general_dict['Name'].append(name)
            general_dict['Hero'].append(hero)
            general_dict['Kill'].append(kd_dict[name][hero]['Kills'])
            general_dict['Death'].append(kd_dict[name][hero]['Deaths'])
            if hero not in ttcu_dict[name]:
                general_dict['TTCU'].append([])
            else:
                general_dict['TTCU'].append(ttcu_dict[name][hero])
            if hero not in ttuu_dict[name]:
                general_dict['TTUU'].append([])
            else:
                general_dict['TTUU'].append(ttuu_dict[name][hero])
            if hero in player_fight_stats[name]['Heroes']:
                general_dict['Fight #'].append(player_fight_stats[name]['Heroes'][hero]['Total'])
                general_dict['Fight Win'].append(player_fight_stats[name]['Heroes'][hero]['Win'])
                general_dict['Fight Lose'].append(player_fight_stats[name]['Heroes'][hero]['Lose'])
                general_dict['First Kill'].append(player_fight_stats[name]['Heroes'][hero]['First Kill'])
                general_dict['First Death'].append(player_fight_stats[name]['Heroes'][hero]['First Death'])
            if hero not in player_fight_stats[name]['Heroes']:
                general_dict['Fight #'].append(0)
                general_dict['Fight Win'].append(0)
                general_dict['Fight Lose'].append(0)
                general_dict['First Kill'].append(0)
                general_dict['First Death'].append(0)

    return general_dict

def create_csvs(date, map, opponent, total_game_time, team1, team2, team1_comps, team2_comps, team1_ttcu, team2_ttcu, team1_ttuu, team2_ttuu, team1_kd, team2_kd, fight_stats, player_fight_stats):
    #Handle Fight CSV
    fight_df = pd.DataFrame.from_dict(fight_stats)
    fight_df.insert(0, 'Date', date)
    fight_df.insert(1, 'Opponent', opponent)

    #Handle Comps CSV
    comps1_df, comps2_df= pd.DataFrame.from_dict(team1_comps), pd.DataFrame.from_dict(team2_comps)
    comps_df = comps1_df.append(comps2_df, ignore_index = True)
    comps_df.insert(0, 'Date', date)
    comps_df.insert(1, 'Map', map)

    #Handle General Information CSV
    general_dict = create_general_dict(team1_ttcu, team2_ttcu, team1_ttuu, team2_ttuu, team1_kd, team2_kd, player_fight_stats)
    general_df = pd.DataFrame.from_dict(general_dict)
    general_df.insert(0, 'Date', date)
    general_df.insert(1, 'Map', map)

    return general_df, comps_df, fight_df

if __name__ == '__main__':
    csv_folder = "csvs/to_csv/"
    stats_folder = "csvs/get_map_stats/"

    for csv in os.listdir(csv_folder):
        csv_path = csv_folder + csv
        stripped_csv_name = csv[:-4]
        date, map, opponent, total_game_time = get_map_info(csv_path)
        team1, team2 = get_rosters(csv_path)
        team1_comps, team2_comps = get_teamcomps(csv_path, map, team1, team2)
        team1_ttcu, team2_ttcu, team1_ttuu, team2_ttuu = get_ttcu_ttuu(csv_path, team1, team2)
        team1_kd, team2_kd = get_kill_deaths(csv_path, team1, team2)
        general_fight_stats, player_fight_stats = get_fight_stats(csv_path, team1, team2, map)
        print_example(csv, date, map, opponent, total_game_time, team1, team2, team1_comps, team2_comps, team1_ttcu, team2_ttcu, team1_kd, team2_kd, general_fight_stats)
        general_df, comps_df, fight_df = create_csvs(date, map, opponent, total_game_time, team1, team2, team1_comps, team2_comps, team1_ttcu, team2_ttcu, team1_ttuu, team2_ttuu, team1_kd, team2_kd, general_fight_stats, player_fight_stats)

        comps_df.to_csv(stats_folder + stripped_csv_name + "+Comps.csv", sep=',')
        print("Created " + stripped_csv_name + "+Comps.csv")
        fight_df.to_csv(stats_folder + stripped_csv_name + "+Fights.csv", sep=',')
        print("Created " + stripped_csv_name + "+Fights.csv")
        general_df.to_csv(stats_folder + stripped_csv_name + "+General.csv", sep=',')
        print("Created " + stripped_csv_name + "+General.csv")
