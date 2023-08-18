def get_schedule(dataset_df, epa_df):
    schedule = dataset_df[
        ['season', 'week', 'home_team', 'away_team', 'home_score', 'away_score']].drop_duplicates().reset_index(
        drop=True).assign(
        home_team_win=lambda x: (x.home_score > x.away_score).astype(int))
    return schedule.merge(epa_df.rename(columns={'team': 'home_team'}), on=['home_team', 'season', 'week']).merge(
        epa_df.rename(columns={'team': 'away_team'}), on=['away_team', 'season', 'week'], suffixes=('_home', '_away'))