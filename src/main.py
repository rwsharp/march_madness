import sys
import os

import numpy as np
import pandas as pd
import sklearn as skl
import re
from collections import Counter

def safe_int(x):
    return int(x) if x.strip() != '' else None


def play_round(round, games, teams):
    for game, info in games.iteritems():
        game_in_round = False

        if round == 'play in':
            if game[0] in ['W', 'X', 'Y', 'Z']:
                game_in_round = True
        elif game[0:2] == round:
            game_in_round = True
        elif game[0] in ['W', 'X', 'Y', 'Z']:
            continue
        elif game[0:2] in ['R1', 'R2', 'R3', 'R4', 'R5', 'R6']:
            continue
        else:
            raise ValueError('Unrecognized round: {} {}'.format(round, game))

        if game_in_round:
            game_pair = tuple(sorted([info['team_0'], info['team_1']]))
            info['winner'] = results[game_pair]
            # print game, '{} vs. {} winner: {}'.format(teams[info['team_0']], teams[info['team_1']], teams[info['winner']])


def update_bracket(round, games, teams):
    for game, info in games.iteritems():
        game_in_round = False

        if game[0:2] == round:
            game_in_round = True
        elif game[0] in ['W', 'X', 'Y', 'Z']:
            continue
        elif game[0:2] in ['R1', 'R2', 'R3', 'R4', 'R5', 'R6']:
            continue
        else:
            raise ValueError('Unrecognized round: {} {}'.format(round, game))

        if game_in_round:
            if info['team_0'] is None:
                info['team_0'] = games[info['parent_0']]['winner']

            if info['team_1'] is None:
                info['team_1'] = games[info['parent_1']]['winner']

            # print game, info


def initialize_bracket(games, seeds, teams):
    # Initialize bracket
    for game, info in games.iteritems():
        if game[0] != 'R':
            info['team_0'] = seeds.get(info['parent_0'])
            info['team_1'] = seeds.get(info['parent_1'])
            # print game, '{} vs. {}'.format(teams[info['team_0']], teams[info['team_1']])

    for game, info in games.iteritems():
        if game[0:2] == 'R1':
            info['team_0'] = seeds.get(info['parent_0'])
            info['team_1'] = seeds.get(info['parent_1'])
            # print game, info


# read tournement data
data_root = 'NCAA Bracket 2017'
history_dir = os.path.join(data_root, 'march-machine-learning-mania-2016-v2')
season = 2015

rankings = dict()
special = ['Rank', 'Team', 'Conf', 'Record', 'Mean', 'Median', 'St.Dev']
string_cols = ['Team', 'Conf', 'Record']
float_cols = ['Mean', 'Median', 'St.Dev']
shift_found = False

with open(os.path.join(data_root, str(season) + '_composite_rankings.csv'), 'r') as input_data:
    for line_number, line in enumerate(input_data):
        team_properties = dict()
        team_rankings = dict()
        if line_number == 0:
            header_str = line.rstrip('\n')
            header = map(lambda s: s.strip().strip(','), line.split())

            N = len(header_str)
            breaks = [N]
            for loc in range(N-1, -1, -1):
                # print loc, header_str[loc], header_str[loc] == ' '
                is_space = (header_str[loc] == ' ')
                if is_space:
                    if loc > 0:
                        next_is_space = (header_str[loc-1] == ' ')
                    else:
                        next_is_space = False

                    if not next_is_space:
                        breaks.append(loc)

            breaks.reverse()
            # print breaks

            # print header_str

        if line.strip() != '' and line.rstrip('\n') != header_str:
            if line_number > -1:
                # print line.rstrip()
                data = line.rstrip('\n')
                team_found = False
                for loc in range(len(breaks)-1):
                    left = breaks[loc]
                    right = breaks[loc + 1]

                    if not shift_found:
                        if header[loc] == 'Rank':
                            right = right - len('Rank,')
                            left = right - 4
                            breaks[loc] = left
                            team_properties[header[loc]] = safe_int(data[left:min(right, len(data))])
                            # print 'a', header[loc], int(data[left:min(right, len(data))])
                        elif header[loc] == 'Team':
                            # print '>>>', left, right, data[left:min(right, len(data))]
                            left = left - len('Rank,')
                            right = left + 17
                            breaks[loc] = left
                            team_properties[header[loc]] = data[left:min(right, len(data))].strip()
                            # print 'b', header[loc], data[left:min(right, len(data))].strip()
                        elif header[loc] == 'Conf':
                            # print '>>>', left, right, data[left:min(right, len(data))]
                            left = left - len('Rank, Team,') + 17
                            right = left + 6
                            breaks[loc] = left
                            team_properties[header[loc]] = data[left:min(right, len(data))].strip()
                            # print 'c', header[loc], data[left:min(right, len(data))].strip()
                        elif header[loc] == 'Record':
                            # print '>>>', left, right, data[left:min(right, len(data))]
                            left = left - len('Rank, Team, Conf,') + 17 + 6
                            right = left + 6
                            breaks[loc] = left
                            breaks[loc + 1] = right
                            shift_found = True
                            team_found = True
                            team_properties[header[loc]] = data[left:min(right, len(data))].strip()
                            # print 'd', header[loc], data[left:min(right, len(data))].strip()
                        else:
                            team_rankings[header[loc]] = safe_int(data[left:min(right, len(data))])
                            # print '*', header[loc], int(data[left:min(right, len(data))])
                    else:
                        if header[loc] == 'Record':
                            team_found = True

                        if not (team_found and header[loc] in ['Team', 'Rank']):
                            if header[loc] in string_cols:
                                team_properties[header[loc]] = data[left:min(right, len(data))].strip()
                                # print '-', header[loc], data[left:min(right, len(data))].strip()
                            elif header[loc] in float_cols:
                                team_properties[header[loc]] = float(data[left:min(right, len(data))])
                                # print '-', header[loc], float(data[left:min(right, len(data))])
                            elif header[loc] != 'Rank':
                                team_rankings[header[loc]] = safe_int(data[left:min(right, len(data))])
                                # print '-', header[loc], safe_int(data[left:min(right, len(data))])
                        else:
                            continue

                team = team_properties['Team']

                rankings[team] = {'properties': team_properties,
                                  'rankings': team_rankings}

                print team, '        \t', rankings[team]


teams = dict()
with open(os.path.join(history_dir, 'Teams.csv'), 'r') as input_data:
    for line_number, line in enumerate(input_data):
        if line_number > 0:
            data = map(lambda s: s.strip(), line.split(','))
            id = int(data[0])
            team = data[1]

            teams[id] = team

games = dict()
with open(os.path.join(history_dir, 'TourneySlots.csv'), 'r') as input_data:
    for line_number, line in enumerate(input_data):
        if line_number > 0:
            data = map(lambda s: s.strip(), line.split(','))
            year = int(data[0])
            game = data[1]
            parent_0 = data[2]
            parent_1 = data[3]

            if year == season:
                games[game] = {'parent_0': parent_0,
                               'parent_1': parent_1,
                               'team_0': None,
                               'team_1': None,
                               'winner': None,
                               'selected_0': None,
                               'selected_1': None}

seeds = dict()
with open(os.path.join(history_dir, 'TourneySeeds.csv'), 'r') as input_data:
    for line_number, line in enumerate(input_data):
        if line_number > 0:
            data = map(lambda s: s.strip(), line.split(','))
            year = int(data[0])
            seed = data[1]
            team_id = int(data[2])

            if year == season:
                seeds[seed] = team_id

results = dict()
with open(os.path.join(history_dir, 'TourneyCompactResults.csv'), 'r') as input_data:
    for line_number, line in enumerate(input_data):
        if line_number > 0:
            data = map(lambda s: s.strip(), line.split(','))
            year = int(data[0])
            winner = int(data[2])
            loser = int(data[4])

            if year == season:
                game_pair = tuple(sorted([winner, loser]))
                results[game_pair] = winner


# initialize
initialize_bracket(games, seeds, teams)

# set actual outcomes
play_round('play in', games, teams)

for r in [1, 2, 3, 4, 5, 6]:
    round = 'R' + str(r)
    update_bracket(round, games, teams)
    print
    play_round(round, games, teams)
    print

# Select bracket
# rank teams
# ranking = dict()
# for game, info in games.iteritems():
#     if game[0:2] == 'R1':
#         ranking[info['team_0']] = np.random.uniform()
#         ranking[info['team_1']] = np.random.uniform()

# rankings = dict()
# with open('NCAA Bracket 2017/rankings_comparison.csv', 'r') as input_data:
#     for line_number, line in enumerate(input_data):
#
#         data = map(lambda s: s.strip(), line.split(','))
#
#         if line_number == 75:
#             header = data[:-1]
#             # print line_number, header
#
#         if line_number >= 77:
#             team = data[0].strip()
#             rankings[team] = {'properties': dict(
#                 [(col, data[i].strip()) for i, col in enumerate(header) if i in range(1, 8)]),
#                               'rankings': dict([(col, safe_int(data[i].strip())) for i, col in enumerate(header) if
#                                                 i in range(8, len(header))])}

rankers = rankings.itervalues().next()['rankings'].keys()

rankers = dict()
for team, info in rankings.iteritems():
    for ranker, rank in info['rankings'].iteritems():
        rankers.setdefault(ranker, {'top': None, 'bottom': None, 'exclude': None})
        if rank is not None:
            if rankers[ranker]['top'] is None:
                rankers[ranker]['top'] = rank
            else:
                rankers[ranker]['top'] = rank if rank < rankers[ranker]['top'] else rankers[ranker]['top']

            if rankers[ranker]['bottom'] is None:
                rankers[ranker]['bottom'] = rank
            else:
                rankers[ranker]['bottom'] = rank if rank > rankers[ranker]['bottom'] else rankers[ranker]['bottom']

            if rankers[ranker]['bottom'] is not None:
                if rankers[ranker]['bottom'] < 100:
                    rankers[ranker]['exclude'] = True
                else:
                    rankers[ranker]['exclude'] = False

included_rankers = list()
for ranker, info in rankers.iteritems():
    if not info['exclude']:
        included_rankers.append(ranker)

print included_rankers


# ranker_set = np.random.choice(included_rankers, 1, replace=False)

for ranker in included_rankers:
    ranker_set = [ranker]

    for trial in range(1):

        votes = list()
        ranking = dict()
        # get all the teams that need to be ranked
        for game, info in games.iteritems():
            if game[0:2] == 'R1':
                # set the teams
                t0 = info['team_0']
                if teams[t0] in rankings:
                    ranking.setdefault(t0, 0.0)
                else:
                    raise ValueError('Unknown Team 0')

                t1 = info['team_1']
                if teams[t1] in rankings:
                    ranking.setdefault(t1, 0.0)
                else:
                    raise ValueError('Unknown Team 1')

        # vote
        for v in range(1):
            # choose a ranker
            ranker = np.random.choice(ranker_set)
            votes.append(ranker)

            # update ranks
            for t in ranking:
                ranking[t] += rankings[teams[t]]['rankings'][ranker]


        # set selections given R1 teams
        for game, info in games.iteritems():
            if game[0:2] == 'R1':
                info['selected_0'] = info['team_0']
                info['selected_1'] = info['team_1']

        # choose winners by rank in subsequent rounds
        for round in ['R2', 'R3', 'R4', 'R5', 'R6']:
            for game, info in games.iteritems():
                if game[0:2] == round:
                    p0 = games[info['parent_0']]
                    p1 = games[info['parent_1']]

                    info['selected_0'] = p0['selected_0'] if ranking[p0['selected_0']] < ranking[p0['selected_1']] else p0['selected_1']
                    info['selected_1'] = p1['selected_0'] if ranking[p1['selected_0']] < ranking[p1['selected_1']] else p1['selected_1']

        # Score bracket:
        score = 0
        points = {'R1': 1,
                  'R2': 2,
                  'R3': 4,
                  'R4': 8,
                  'R5': 16,
                  'R6': 32}

        for game, info in games.iteritems():
            if game[0] != 'R':
                round = 'play in'
                continue
            else:
                round = game[0:2]
                selected_winner = info['selected_0'] if ranking[info['selected_0']] < ranking[info['selected_1']] else info['selected_1']
                if info['winner'] == selected_winner:
                    p = points[round]
                else:
                    p = 0

                score += p

                # if ranker == 'RT':
                #     print '{}, {} ({}) vs. {} ({}), selected: {}, winner: {}, points: {}'.format(round, \
                #                                                                                  teams[info['team_0']], \
                #                                                                                  ranking[info['team_0']], \
                #                                                                                  teams[info['team_1']], \
                #                                                                                  ranking[info['team_1']], \
                #                                                                                  teams[selected_winner], \
                #                                                                                  teams[info['winner']], \
                #                                                                                  p)

        print '{}: {}'.format([(ranker, Counter(votes)[ranker]) for ranker in ranker_set], score)



sys.exit(1)




teams = dict()
header = True
with open('NCAA Bracket 2017/LRMC.csv', 'r') as input_data:
    for line_number, line in enumerate(input_data):
        #print line_number, line.strip()

        data = map(lambda s: s.strip(), line.split(','))

        if data == ['' for i in range(31)]:
            header = not header
            continue

        if not header:
            if line_number %2 == 1:
                teams[data[2]] =  {'LRMC': data[1],
                                   'conference': data[3],
                                   'record': data[5],
                                   'mean opponent rank': data[6]}



        # debug code - make sure parsing won't choke on a line with extra or missing separators
        if len(line.split(',')) != 31:
            print 'ERROR - wrong number of commas'
            print line_number
            print line
            break