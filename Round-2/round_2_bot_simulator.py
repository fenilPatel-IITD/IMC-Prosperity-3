from abc import abstractmethod
import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


NUMBER_OF_BOTS = 4000  # Total number of bots
SKEW_FACTOR = 0.01  # Skew factor for profit calculation
RE_THINK_ROUNDS = 1000  # Number of rounds for rethinking

class Container:
    def __init__(self, big_number, inhabitants):
        self.big_number = big_number  # BIG number on the container
        self.coins = 10000 * big_number  # Coins in the container
        self.inhabitants = inhabitants  # Total number of inhabitants
        self.chosen_by = {i: 0 for i in range(1, NUMBER_OF_BOTS+1)}  # Track how many bots choose this container
    
    def choose(self, bot_id):
        # print(f"Bot {bot_id} chose container with {self.coins} coins")
        self.chosen_by[bot_id] = 1  # Increment the count for this bot
    def dechoose(self, bot_id):
        self.chosen_by[bot_id] = 0
    
    @abstractmethod
    def profit(self, skew_factor = 1):
        # print(self.chosen_by)
        total_chosen = sum(self.chosen_by.values())*100*skew_factor / NUMBER_OF_BOTS
        # print("Total chosen:", total_chosen)
        profit = self.coins / (self.inhabitants + total_chosen)
        return profit




class TreasureGame:
    def __init__(self, containers, skew_factor=0.1):
        self.containers = containers  # List of BIG numbers on containers
        self.skew_factor = skew_factor  # Factor to skew the data for bots

    def calculate_profit(self, container):
        return Container.profit(container,0)

    def simulate(self, bots):
        results = []
        for bot in bots:
            bot.choose_containers(self.containers, self.skew_factor)
            total_profit = Container.profit(bot.chosen, 0) # Calculate profit for the chosen container
            # print(f"Bot {bot.name} chose container with total {total_profit} coins")
            # if len(chosen_containers) == 2:
            #     total_profit -= 50000  # Fee for choosing two containers
            results.append((bot.name, total_profit))
        return sorted(results, key=lambda x: x[1], reverse=True)

    def animate_choices(self, bots):
        fig, ax = plt.subplots()
        container_labels = [str(c.big_number) for c in self.containers]
        bar_container = ax.bar(container_labels, [0] * len(self.containers))

        def update(frame):
            for container in self.containers:
                container.chosen_by = {i: 0 for i in range(1, NUMBER_OF_BOTS+1)}
            for bot in bots[:frame * 100]:
                bot.choose_containers(self.containers, self.skew_factor)

            counts = [sum(c.chosen_by.values()) / NUMBER_OF_BOTS * 100 for c in self.containers]
            for bar, count in zip(bar_container, counts):
                bar.set_height(count)

            ax.set_title(f"Bots Choosing Containers - Frame {frame}")
            ax.set_ylim(0, 100)
            ax.set_ylabel("Percentage of Bots (%)")

        ani = FuncAnimation(fig, update, frames=range(1, 500), repeat=False, interval=10)
        plt.show()


class Bot:
    def __init__(self, name, rethink_list):
        self.name = name
        self.rethink_list = rethink_list  # List of 100 values indicating if the bot rethinks its choice
        self.chosen = None  # Chosen container

    def choose_containers(self, containers, skew_factor):
        
        x = sorted([(Container.profit(c, 1 + random.uniform(-skew_factor, skew_factor) + random.randrange(-500, 2000)/1000 ), c) for c in containers], reverse=True, key=lambda x: x[0])
        if self.chosen is not None:
            self.chosen.dechoose(self.name)
        x[0][1].choose(self.name)  # Choose the first container
        self.chosen = x[0][1]  # Store the chosen container
        return x[0][1] #, x[1][1]  # Return the two containers with the highest profit
    def get_chosen(self):
        return self.chosen

# Example usage
if __name__ == "__main__":
    containers = [
        Container(10, 1),
        Container(20, 2),
        Container(31, 2),
        Container(37, 3),
        Container(50, 4),
        Container(17, 1),
        Container(73, 4),
        Container(80, 6),
        Container(89, 8),
        Container(90, 10)
    ]  # 10 containers with random BIG numbers

    # Create bots with random rethink lists
    bots = [
        Bot(i, [random.choice([True, False]) for _ in range(RE_THINK_ROUNDS)]) for i in range(1, NUMBER_OF_BOTS+1)
    ]

    game = TreasureGame(containers, SKEW_FACTOR)
    results = game.simulate(bots)

    print("Game Results:")
    # for bot in bots:
    #     print(f"{bot.name}: {bot.chosen.big_number} coins")
    sorted_containers = sorted(containers, key=lambda c: c.profit(), reverse=True)
    for container in sorted_containers:
        print(f"Container {container.big_number}: {container.coins} coins | chosen by {sum(container.chosen_by.values())} bots | profit: {container.profit()}")
    
    game.animate_choices(bots)