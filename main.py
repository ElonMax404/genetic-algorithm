import pygame
from typing_extensions import Self
from math import sqrt, exp
import random
import numpy as np # only gonna use it for softmax
import copy

class Evolution:
    def __init__(self, n_games: int) -> Self:
        self.games: list[Game] = [] 
        self.generation = 1
        for x in range(int(sqrt(n_games))):
            for y in range(int(sqrt(n_games))):
                field_wh = screen.get_size()[0]/int(sqrt(n_games))
                game = Game(coordinates=pygame.Vector2((x*100, y*100)), field_size=pygame.Vector2((field_wh, field_wh)))
                self.games.append(game)
    def get_n_active(self) -> int:
        n_active = 0
        for game in self.games:
            if game.running:
                n_active += 1
        return n_active
    def update(self):
        for game in simulation.games:
            game.update()
            game.draw(screen)
        
        if self.get_n_active() == 0:
            fitness_scores = [game.agent.fitness for game in simulation.games]
            best_indices = Evolution.get_best(fitness_scores)
            best_agents = [simulation.games[i].agent for i in best_indices]
            print(f"Generation {self.generation}, best agent: {best_agents[0].fitness}")
            for i, game in enumerate(simulation.games):
                game.agent = Agent(
                    position=game.coordinates + (game.field_size / 2), 
                    size=game.field_size.x / 10, 
                    grid_coords=game.coordinates, 
                    field_size=game.field_size, 
                    user_controlled=game.agent.user_controlled
                )
                if i < len(best_agents):
                    game.agent.brain = copy.deepcopy(best_agents[i].brain)
                else:
                    best_one = random.choice(best_agents)
                    game.agent.brain = copy.deepcopy(best_one.brain)
                    game.agent.brain.mutate(chance=30)
                
                game.agent.spawn_goal()
                game.n_ticks = 0
                game.running = True
            self.generation += 1



            
    def get_best(scores: list[float]) -> list[int]:
        sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return sorted_indices[:3]  # Return indices of top 5 agents

    

class Layer:
    def __init__(self, n_inputs: int, n_neurons: int, activation: str) -> Self:
        self.weights: list[list[float]] = []
        self.biases: list[float] = []
        if activation == "relu":
            self.activationFn = self.activationRELU
        elif activation == "softmax":
            self.activationFn = self.activationSoftmax
        else:
            raise ValueError("Undefined activation function.")
        for _ in range(n_neurons):
            neuron_weight = []
            for _ in range(n_inputs):
                neuron_weight.append(random.uniform(-1, 1))
            self.weights.append(neuron_weight)
            self.biases.append(random.uniform(-1, 1))
    def activationRELU(self, inputs: list[float]):
        outputs = []
        for input in inputs:
            if input < 0:
                outputs.append(0)
            else:
                outputs.append(input)
        return outputs
    def activationSoftmax(self, x): # shamelessly stolen from the internet
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)
    def forward(self, inputs: list[float]) -> list[float]:
        layer_outputs = []
        for neuron_weights, bias in zip(self.weights, self.biases):
            neuron_output = 0
            for input, weight in zip(inputs, neuron_weights):
                neuron_output += input*weight
            neuron_output += bias
            layer_outputs.append(neuron_output)
        layer_outputs = self.activationFn(layer_outputs)
        return layer_outputs
    

class AgentNeuralNetwork:
    def __init__(self) -> Self:
        self.layers: list[Layer] = []
        self.layers.append(Layer(6, 6, "relu"))
        self.layers.append(Layer(6, 4, "softmax"))
    def forward(self, inputs: list[float]) -> list[float]:
        layer_inputs = inputs
        for layer in self.layers:
            layer_output = layer.forward(layer_inputs)
            layer_inputs = layer_output
        return layer_output
    def mutate(self, chance: int):
        for layer in self.layers:
            for i in range(len(layer.weights)):
                for j in range(len(layer.weights[i])):
                    if random.randint(0, 100) < chance:
                        layer.weights[i][j] += random.uniform(-0.5, 0.5)
            for i in range(len(layer.biases)):
                if random.randint(0, 100) < chance:
                    layer.biases[i] += random.uniform(-0.5, 0.5)

            
class Agent:
    def __init__(self, position: pygame.Vector2, size: int, grid_coords: pygame.Vector2, field_size: pygame.Vector2, user_controlled: bool = False) -> Self:
        self.grid_coords = grid_coords
        self.user_controlled = user_controlled
        self.position = position
        self.velocity = pygame.Vector2((0, 0))
        self.size = size
        self.brain = AgentNeuralNetwork()
        self.field_size = field_size
        self.spawn_goal()
        self.fitness = 10
    def global_to_relative(self, globl: pygame.Vector2) -> pygame.Vector2:

        return globl - self.grid_coords
    
    def spawn_goal(self):

        min_x = self.grid_coords.x + 20
        max_x = self.grid_coords.x + self.field_size.x - 20
        min_y = self.grid_coords.y + 20
        max_y = self.grid_coords.y + self.field_size.y - 20
        self.goal_coordinates = pygame.Vector2((random.randint(min_x, max_x), random.randint(min_y, max_y)))
        # self.goal_coordinates = pygame.Vector2((self.grid_coords.x + 30, self.grid_coords.y + 30))
    def update(self):
        rel_pos = self.global_to_relative(self.position)
        posX = rel_pos.x
        posY = rel_pos.y
        velX = self.velocity.x
        velY = self.velocity.y
        goal = self.global_to_relative(self.goal_coordinates)
        goalX = goal.x
        goalY = goal.y
        inputs = [posX, posY, velX, velY, goalX, goalY]
        decision = self.brain.forward(inputs)
        biggest_i = 0
        biggest_output = 0
        for i, output in enumerate(decision):
            if output > biggest_output:
                biggest_output = output
                biggest_i = i
        
        if biggest_i == 0:
            self.velocity.y += -2
        elif biggest_i == 1:
            self.velocity.x += -0.1
        elif biggest_i == 2:
            self.velocity.x += 0.1
        elif biggest_i == 3:
            pass
        distance = (self.position - self.goal_coordinates).magnitude()
        self.fitness += (1 / (distance + 1e-5) )
        self.fitness += abs(self.velocity.x) * 0.1
        


class Game:
    def __init__(self, coordinates: pygame.Vector2, field_size: pygame.Vector2, user_controlled: bool = False, gravity: float = 0.5, air_resistance: float = 0.99) -> Self:
        self.field_size = field_size
        self.agent = Agent(position=coordinates+(field_size/2), size=field_size.x/10, grid_coords=coordinates, field_size=field_size, user_controlled=user_controlled)
        self.gravity = gravity
        self.air_resistance = air_resistance
        self.coordinates = coordinates
        self.running = True
        self.n_ticks = 0

    def update(self):
        if self.running:
            self.agent.velocity.y += self.gravity
            self.agent.velocity.x *= self.air_resistance
            if abs(self.agent.velocity.x) <= 0.0001:
                self.agent.velocity.x = 0
            self.agent.position += self.agent.velocity
            if self.check_goal():
                self.agent.fitness += 100
                self.n_ticks = 0
                self.agent.spawn_goal()
            if self.check_game_over():
                self.game_over()
            self.agent.update()
            self.n_ticks += 1
            


    def check_game_over(self) -> bool:
        if (self.agent.position.x + self.agent.size >= self.coordinates.x + self.field_size.x or
            self.agent.position.x <= self.coordinates.x or
            self.agent.position.y + self.agent.size >= self.coordinates.y + self.field_size.y or
            self.agent.position.y <= self.coordinates.y):
                self.agent.fitness += -100
                return True
        elif self.n_ticks > 500:
            return True
        return False
    def handle_input(self, input_event: pygame.event.Event):
        key = input_event.key
        if key == pygame.K_SPACE:
            self.agent.velocity.y += -2
        elif key == pygame.K_a:
            self.agent.velocity.x += -0.5
        elif key == pygame.K_d:
            self.agent.velocity.x += 0.5
    def draw(self, screen: pygame.Surface):
        pygame.draw.rect(screen, "black", (self.coordinates, (self.coordinates.x + self.field_size.x+1, self.coordinates.y + self.field_size.y+1)), width=1)
        if self.running:
            pygame.draw.rect(screen, "green", (self.agent.position, self.field_size/10))
            pygame.draw.rect(screen, "red", (self.agent.goal_coordinates, self.field_size/10))
        
    def game_over(self):
        self.running = False
    
    def check_goal(self) -> bool:
        if (self.agent.goal_coordinates - self.agent.position).magnitude() < 10:
            return True
        return False


pygame.init()
screen = pygame.display.set_mode((800, 800))
pygame.display.set_caption("Genetic algorithm")
clock = pygame.time.Clock()


# game = Game(coordinates=pygame.Vector2((0, 0)), field_size=pygame.Vector2((800, 800)), user_controlled=True)
# games.append(game)

simulation = Evolution(64)

running = True

while running:

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
            for game in simulation.games:
                if game.agent.user_controlled:
                    game.handle_input(event)

        
    screen.fill("grey")
    simulation.update()

    pygame.display.flip()
    clock.tick(200)

pygame.quit()