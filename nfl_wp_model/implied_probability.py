def favorite_implied_probability(team, odds):
    result = (-1 * odds) / ((-1 * odds) + 100)
    print(f"Favorite {team} likelihood of winning: {round(result * 100, 2)}%")
    return result

def underdog_implied_probability(team, odds):
    result = 100 / (odds + 100)
    print(f"Underdog {team} likelihood of winning: {round(result * 100, 2)}%")
    return result


def calculate_house_edge(wp_favorite, wp_underdog):
    result = (((wp_favorite + wp_underdog) - 1) * 100)
    return f"House edge: {round(result, 2)}%"

def calculate_player_edge(model_probability, implied_probability):
    result = (model_probability - implied_probability) * 100
    return f"Player Edge: {round(result,2)}%"