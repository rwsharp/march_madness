import sys
import os
import glob

import numpy as np
import pandas as pd
import sklearn as skl
import re
from collections import Counter

def safe_int(x):
    return int(x) if x.strip() != '' else None


def play_round(round, games, results, teams):
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

        try:
            if game_in_round:
                game_pair = tuple(sorted([info['team_0'], info['team_1']]))
                info['winner'] = results[game_pair]
                # print game, '{} vs. {} winner: {}'.format(teams[info['team_0']], teams[info['team_1']], teams[info['winner']])
        except:
            print round
            raise

def play_ranked_round(round, games, bracket, teams):
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

        try:
            if game_in_round:
                game_pair = tuple(sorted([info['team_0'], info['team_1']]))
                if bracket[game_pair[0]] < bracket[game_pair[1]]:
                    info['winner'] = game_pair[0]
                else:
                    info['winner'] = game_pair[1]

                # if teams[info['winner']] == 'Northwestern':
                #     print game, '{} vs. {} winner: {}'.format(teams[info['team_0']], teams[info['team_1']], teams[info['winner']])
        except:
            print round
            raise


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

def read_rankings(data_root, season):
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
                for loc in range(N - 1, -1, -1):
                    # print loc, header_str[loc], header_str[loc] == ' '
                    is_space = (header_str[loc] == ' ')
                    if is_space:
                        if loc > 0:
                            next_is_space = (header_str[loc - 1] == ' ')
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
                    for loc in range(len(breaks) - 1):
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

                    # print team, '        \t', rankings[team]

    return rankings


def read_teams(history_dir):
    teams = dict()
    with open(os.path.join(history_dir, 'Teams.csv'), 'r') as input_data:
        for line_number, line in enumerate(input_data):
            if line_number > 0:
                data = map(lambda s: s.strip(), line.split(','))
                id = int(data[0])
                team = data[1]

                teams[id] = team

    return teams


def read_games(history_dir, season):
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

    return games


def read_seeds(history_dir, season):
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

    return seeds


def read_results(history_dir, season):
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

    return results


def get_rankers(rankings):
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

    return included_rankers


def get_mmad_teams(games):
    mmad_teams = list()
    for game, info in games.iteritems():
        if game[0:2] == 'R1':
            mmad_teams.append(info['team_0'])
            mmad_teams.append(info['team_1'])

    return mmad_teams


def get_bracket(ranker, rankings, mmad_teams, teams):
    bracket = dict()
    for team in mmad_teams:
        bracket[team] = rankings[teams[team]]['rankings'][ranker]

    return bracket


def get_voter_bracket(ranker_set, n, rankings, mmad_teams, teams):
    bracket = dict()
    votes = list()

    # Each team is ranked n times by n randomly selected rankers from the ranker_set.
    # final ranking is normalized by n to make brackets comparable if the ranker_set changes size.
    for team in mmad_teams:
        for vote in range(n):
            ranker = np.random.choice(ranker_set)
            votes.append(ranker)
            bracket[team] = rankings[teams[team]]['rankings'][ranker]
        bracket[team] = bracket[team]/float(n)

    return bracket


def get_randomized_bracket(ranker, noise_magnitude, rankings, mmad_teams, teams):
    bracket = dict()

    # Each team is ranked n times by n randomly selected rankers from the ranker_set.
    # final ranking is normalized by n to make brackets comparable if the ranker_set changes size.
    for team in mmad_teams:
        bracket[team] = rankings[teams[team]]['rankings'][ranker] + np.random.normal(loc=0, scale=noise_magnitude)

    return bracket


def get_mascot_randomized_bracket(ranker, mascot_noise, rankings, mmad_teams, teams):
    bracket = dict()

    # Each team is ranked n times by n randomly selected rankers from the ranker_set.
    # final ranking is normalized by n to make brackets comparable if the ranker_set changes size.
    for team in mmad_teams:
        noise = mascot_noise[team]
        bracket[team] = rankings[teams[team]]['rankings'][ranker] + noise

    return bracket


def set_selections(games, ranking):
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


def score_bracket(games, ranking):

    points = {'R1': 1,
              'R2': 2,
              'R3': 4,
              'R4': 8,
              'R5': 16,
              'R6': 32}

    score = 0
    possible = 0

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
            possible += points[round]

    return score, possible


def compute_single_ranker_records(season_range_low, season_range_high, rankers, rankings, mmad_teams, teams, games):
    # set brackets
    season_range = range(season_range_low, season_range_high + 1)

    records = dict()
    for season in season_range:
        for ranker in rankers[season]:
            records.setdefault(ranker, dict())

            bracket = get_bracket(ranker, rankings[season], mmad_teams[season], teams)
            set_selections(games[season], bracket)
            score, possible = score_bracket(games[season], bracket)

            records[ranker][season] = score
            # print '{}: {} ({})'.format(ranker, score, possible)

    delimiter = '|'
    header = ['Ranker'] + [str(season) for season in season_range]
    print delimiter.join(header)
    for ranker, record in records.iteritems():
        print_data = [ranker] + [record.get(season, '') for season in season_range]
        print delimiter.join(map(str, print_data))


def compute_pair_ranker_records(season_range_low, season_range_high, votes, rankers, rankings, mmad_teams, teams, games):
    # set brackets
    season_range = range(season_range_low, season_range_high + 1)

    records = dict()

    for season in season_range:
        print season
        n_rankers = len(rankers[season])
        n_pairs = n_rankers*(n_rankers - 1)/2 - n_rankers
        ranker_pairs = set()
        pair_n = 0
        for ranker_1 in rankers[season]:
            for ranker_2 in rankers[season]:
                if ranker_1 != ranker_2:
                    ranker_pair = tuple(sorted([ranker_1, ranker_2]))
                    if ranker_pair not in ranker_pairs:
                        ranker_pairs.add(ranker_pair)
                        pair_n += 1
                        # print '     {} ({}/{}) {}'.format(ranker_pair, pair_n, n_pairs, n_rankers)
                        records.setdefault(ranker_pair, dict())

                        bracket = get_voter_bracket(ranker_pair, votes, rankings[season], mmad_teams[season], teams)
                        set_selections(games[season], bracket)
                        score, possible = score_bracket(games[season], bracket)

                        records[ranker_pair][season] = score

    return records

def compute_random_ranker_score(votes, ranker_set, rankings, mmad_teams, teams, games):
    bracket = get_voter_bracket(ranker_set, votes, rankings, mmad_teams, teams)
    set_selections(games, bracket)
    score, possible = score_bracket(games, bracket)

    return score


def randomized_ranker_score(ranker, noise_magnitude, rankings, mmad_teams, teams, games):
    bracket = get_randomized_bracket(ranker, noise_magnitude, rankings, mmad_teams, teams)
    set_selectiofns(games, bracket)
    score, possible = score_bracket(games, bracket)

    return score


def main():
    data_root = 'NCAA Bracket 2017'
    history_dir = os.path.join(data_root, 'march-machine-learning-mania-2017')
    rankings = dict()
    rankers = dict()
    games = dict()
    seeds = dict()
    results = dict()
    mmad_teams = dict()

    season_range_low = 2010
    season_range_high = 2015
    season_range = range(season_range_low, season_range_high + 1)

    teams = read_teams(history_dir)

    ##################
    # read data
    ##################

    # for season in season_range:
    #     rankings[season] = read_rankings(data_root, season)
    #     rankers[season] = get_rankers(rankings[season])
    #
    #     games[season] = read_games(history_dir, season)
    #     seeds[season] = read_seeds(history_dir, season)
    #     results[season] = read_results(history_dir, season)
    #
    #     # initialize
    #     initialize_bracket(games[season], seeds[season], teams)
    #
    #     # set actual outcomes
    #     play_round('play in', games[season], results[season], teams)
    #
    #     for r in [1, 2, 3, 4, 5, 6]:
    #         round = 'R' + str(r)
    #         update_bracket(round, games[season], teams)
    #         play_round(round, games[season], results[season], teams)
    #
    #     # get the list of 64 teams in March Madness
    #     mmad_teams[season] = get_mmad_teams(games[season])


    # create and score brackets
    # compute_single_ranker_records(season_range_low, season_range_high, rankers, rankings, mmad_teams, teams, games)

    # votes = 1
    # records = compute_pair_ranker_records(season_range_low, season_range_high, votes, rankers, rankings, mmad_teams, teams, games)
    #
    # stats = dict()
    # for ranker, record in records.iteritems():
    #     stats.setdefault(ranker, dict())
    #     data = [record[season] for season in season_range if record.get(season) is not None]
    #     stats[ranker]['n']    = len(data)
    #     stats[ranker]['min']  = np.min(data)
    #     stats[ranker]['max']  = np.max(data)
    #     stats[ranker]['mean'] = np.mean(data)
    #
    # # lets only consider rankers that have been around for the entire set of seasons and make them rankable by mean score
    # select_rankers = list()
    # for ranker, info in stats.iteritems():
    #     if info['n'] == len(season_range):
    #         select_rankers.append((info['mean'], ranker))
    #
    # delimiter = '|'
    # header = ['Ranker 1', 'Ranker 2', 'mean score']
    # print delimiter.join(header)
    #
    # for mean_score, ranker_pair in sorted(select_rankers):
    #     print_data = [ranker_pair[0], ranker_pair[1], mean_score]
    #     print delimiter.join(map(str, print_data))


    ####################################
    # Run single ranker + gaussian noise
    ####################################
    #
    # select_rankers = ['SAG', 'DOK', 'BOB', 'WLK', 'KPK', 'PIG', 'STH', 'POM', 'SE', 'RTH', 'WIL', 'CPA', 'MAS', 'PGH',
    #                   'DC', 'CNG', 'MOR', 'DCI', 'LMC', 'KRA', 'WOB', 'DOL', 'CPR', 'BIH', 'RT', 'REW', 'WOL', 'NOL',
    #                   'COL', 'SPW', 'RTR', 'RPI']
    #
    # select_rankers_2 = ['SAG', 'DOK', 'BOB', 'WLK', 'KPK', 'PIG', 'STH', 'POM', 'SE', 'RTH', 'WIL', 'CPA', 'MAS', 'PGH',
    #                   'DC', 'CNG', 'MOR', 'DCI', 'LMC', 'KRA', 'WOB', 'DOL', 'CPR', 'BIH', 'RT', 'REW', 'WOL', 'NOL',
    #                   'COL', 'SPW', 'RTR']
    #
    # for noise_magnitude in [2, 4, 6, 8, 10, 12, 14]:
    #     n_trials = 1000
    #     delimiter = '|'
    #     # for ranker in select_rankers:
    #     stats = dict()
    #
    #     for ranker in ['SAG', 'LMC', 'RPI']:
    #         stats.setdefault(ranker, dict())
    #         for season in season_range:
    #             stats[ranker].setdefault(season, dict())
    #             stats[ranker][season].setdefault('data', list())
    #             for t in range(n_trials):
    #                 score = randomized_ranker_score(ranker, noise_magnitude, rankings[season], mmad_teams[season], teams, games[season])
    #                 plain_score = compute_random_ranker_score(1, [ranker], rankings[season], mmad_teams[season], teams, games[season])
    #                 stats[ranker][season]['data'].append(score - plain_score)
    #
    #     for ranker, info in sorted(stats.iteritems()):
    #         p90 = list()
    #         for season, st in sorted(info.iteritems()):
    #             p90.append(np.percentile(st['data'], 90))
    #             # print noise_magnitude, ranker, season, np.median(st['data'])
    #         print noise_magnitude, ranker, np.mean(p90)

                # with open(os.path.join(data_root, 'random_brackets.csv'), 'w') as output_file:
    #     delimiter = '|'
    #     header = ['votes', 'season', 'score', 'RPI score', 'lift over RPI']
    #     print >> output_file, delimiter.join(header)
    #
    #     n_trials = 1000
    #     n_rankers = 2
    #     for votes in [n_rankers]:
    #         print 'votes:', votes
    #         for t in range(n_trials):
    #             #n_rankers = 2
    #             #ranker_set = sorted(np.random.choice(select_rankers, n_rankers, replace=False))
    #
    #             season = season_range[np.random.randint(0, len(season_range))]
    #             #ranker_set = select_rankers[:n_rankers]
    #             ranker_set = ['KPK', 'PIG']
    #             rpi_score = compute_random_ranker_score(1, ['RPI'], rankings[season], mmad_teams[season], teams, games[season])
    #             sag_score = compute_random_ranker_score(1, ['SAG'], rankings[season], mmad_teams[season], teams, games[season])
    #             score = compute_random_ranker_score(votes, ranker_set, rankings[season], mmad_teams[season], teams, games[season])
    #             print_data = [votes, season] + [score, rpi_score, sag_score, (score - rpi_score)/float(rpi_score)]
    #
    #             # season = season_range[np.random.randint(0, len(season_range))]
    #             # ranker_set = sorted(['DOK', 'RPI'])
    #             # rpi_score = compute_random_ranker_score(1, ['RPI'], rankings[season], mmad_teams[season], teams, games[season])
    #             # score = compute_random_ranker_score(votes, ranker_set, rankings[season], mmad_teams[season], teams, games[season])
    #             # print_data += [votes, season] + list(ranker_set) + [score, rpi_score, (score-rpi_score)/float(rpi_score)]
    #
    #             print >> output_file, delimiter.join(map(str, print_data))
    #             if t % 500 == 0:
    #                 print t, delimiter.join(map(str, print_data))


    ####################################
    # Generate Random brackets
    ####################################

    noise_magnitude = 4.0
    trials = 100
    season = 2017
    delimiter = '|'

    rankings[season] = read_rankings(data_root, season)
    rankers[season] = get_rankers(rankings[season])

    games[season] = read_games(history_dir, season)
    seeds[season] = read_seeds(history_dir, season)
    results[season] = read_results(history_dir, season)

    # initialize
    initialize_bracket(games[season], seeds[season], teams)

    reverse_teams = dict([(team, id) for id, team in teams.iteritems()])

    # print mmad_teams_2017
    #
    # print teams[1344], teams[1425], teams[1291], teams[1309], teams[1300], teams[1413], teams[1243], teams[1448]

    # set actual outcomes of play-in games
    # slots
    # 2017, W11, W11a, W11b
    # 2017, W16, W16a, W16b
    # 2017, Y16, Y16a, Y16b
    # 2017, Z11, Z11a, Z11b

    # seeds
    # 2017, W11a, 1344 - Providence
    # 2017, W11b, 1425 - USC
    # 2017, W16a, 1291 - Mt St Mary's
    # 2017, W16b, 1309 - New Orleans
    # 2017, Y16a, 1300 - NC Central
    # 2017, Y16b, 1413 - UC Davis
    # 2017, Z11a, 1243 - Kansas St
    # 2017, Z11b, 1448 - Wake Forest

    # results
    # 2017, 0-2, 1344, 71, 1425, 75, N, 0 --> W11 winner = 1425
    # 2017, 0-2, 1291, 67, 1309, 66, N, 0 --> W16 winner = 1291
    # 2017, 0-1, 1300, 63, 1413, 67, N, 0 --> Y16 winner = 1413
    # 2017, 0-1, 1243, 95, 1448, 88, N, 0 --> Z11 winner = 1243

    # games[game(W11, e.g.)]['winner'] = team_id


    # for g in ['W11', 'W16', 'Y16', 'Z11']:
    #     print g, teams[games[2017][g]['winner']]

    # get the list of 64 teams in March Madness
    mmad_teams[season] = get_mmad_teams(games[season])

    select_rankers = ['SAG', 'DOK', 'WLK', 'KPK', 'PIG', 'STH', 'POM', 'RTH', 'WIL', 'MAS', 'PGH',
                      'DC', 'CNG', 'MOR', 'DCI', 'LMC', 'KRA', 'WOB', 'DOL', 'BIH', 'RT', 'REW', 'WOL', 'NOL',
                      'COL', 'SPW', 'RPI']

    # select_rankers = ['RPI']

    stats = {'final four': dict(),
             'final two': dict(),
             'champion': list()}

    trial = -1
    for ranker in select_rankers:
        mascot_file = os.path.join(data_root, 'mascot_rank.5_0.0.csv')
        # for mascot_file in glob.glob(os.path.join(data_root, 'mascot_rank.5_0.*.csv')):
        #     trial += 1
        #     print trial
        mmad_teams_2017 = list()
        mascot_noise = dict()
        with open(mascot_file, 'r') as input_file:
            for line in input_file:
                team, rank, noise = line.split(delimiter)
                noise = float(noise)

                if team not in reverse_teams:
                    print team, team in reverse_teams
                    raise ValueError('ERROR - could not find team.')
                mmad_teams_2017.append(reverse_teams[team])
                mascot_noise[reverse_teams[team]] = noise

        for trial in range(trials):
            with open(os.path.join(data_root, 'brackets/bracket.ranker_{}.stdev_{}.{}.csv'.format(ranker, noise_magnitude, trial)), 'w') as output_file:
                games[season] = read_games(history_dir, season)
                seeds[season] = read_seeds(history_dir, season)
                results[season] = read_results(history_dir, season)

                initialize_bracket(games[season], seeds[season], teams)

                # set the play-in results
                games[2017]['W11']['winner'] = 1425
                games[2017]['W16']['winner'] = 1291
                games[2017]['Y16']['winner'] = 1413
                games[2017]['Z11']['winner'] = 1243

                bracket = get_randomized_bracket(ranker, noise_magnitude, rankings[season], mmad_teams_2017, teams)
                # bracket = get_mascot_randomized_bracket(ranker, mascot_noise, rankings[season], mmad_teams_2017, teams)
                for team, rank in bracket.iteritems():
                    print >> output_file, team, rank

                for r in [1, 2, 3, 4, 5, 6]:
                    round = 'R' + str(r)
                    update_bracket(round, games[season], teams)
                    play_ranked_round(round, games[season], bracket, teams)

                # print ranker
                for game, round in [('R5WX', 'Final Four'), ('R5YZ','Final Four'), ('R6CH', 'Championship')]:
                    # print '{}\t({} vs. {})\twinner: {}'.format(round,
                    #                                            teams[games[season][game]['team_0']],
                    #                                            teams[games[season][game]['team_1']],
                    #                                            teams[games[season][game]['winner']])
                    if round == 'Final Four':
                        stats['final four'].setdefault(trial, list())
                        stats['final four'][trial].extend([teams[games[season][game]['team_0']], teams[games[season][game]['team_1']]])
                    elif round == 'Championship':
                        stats['final two'].setdefault(trial, list())
                        stats['final two'][trial].extend([teams[games[season][game]['team_0']], teams[games[season][game]['team_1']]])
                        stats['champion'].append(teams[games[season][game]['winner']])

                    # print

    Z = len(stats['final four'])
    ff_teams = list()
    for trial in stats['final four']:
        ff_teams.extend(stats['final four'][trial])
    ff_teams = set(ff_teams)
    for team in ff_teams:
        n = sum([1 for trial, ff in stats['final four'].iteritems() if team in ff])
        pct = n/float(Z)
        print '{}|{}|{}|{}|{}'.format('Final Four', team, n, Z, np.round(100*pct, 2))
    print


    Z = len(stats['final two'])
    ff_teams = list()
    for trial in stats['final two']:
        ff_teams.extend(stats['final two'][trial])
    ff_teams = set(ff_teams)
    for team in ff_teams:
        n = sum([1 for trial, ff in stats['final two'].iteritems() if team in ff])
        pct = n/float(Z)
        print '{}|{}|{}|{}|{}'.format('Final', team, n, Z, np.round(100*pct, 2))
    print

    Z = sum([n for team, n in Counter(stats['champion']).most_common()])
    for team, pct in [(team, n/float(Z)) for team, n in Counter(stats['champion']).most_common()]:
        print '{}|{}|{}'.format('Champion', team, np.round(100*pct, 2))



    # for ranker in ['SAG', 'LMC', 'RPI']:
    #     # bracket = get_randomized_bracket(ranker, noise_magnitude, rankings[season], mmad_teams_2017, teams)
    #     bracket = get_mascot_randomized_bracket(ranker, mascot_noise, rankings[season], mmad_teams_2017, teams)
    #     # for rank, team_id in sorted([(rank, team_id) for team_id, rank in sorted(bracket.iteritems())]):
    #     #     print ranker, teams[team_id], rank
    #     # print


if __name__ == '__main__':
    main()