import pygame
from typing_extensions import Self
from math import sqrt

class Agent:
    def __init__(self, position: pygame.Vector2, size: int, user_controlled: bool = False):
        self.user_controlled = user_controlled
        self.position = position
        self.velocity = pygame.Vector2((0, 0))
        self.size = size


class Game:
    def __init__(self, coordinates: pygame.Vector2, field_size: pygame.Vector2, user_controlled: bool = False, gravity: float = 0.1, air_resistance: float = 0.99) -> Self:
        self.field_size = field_size
        self.agent = Agent(position=coordinates+(field_size/2), size=field_size.x/10, user_controlled=user_controlled)
        self.gravity = gravity
        self.air_resistance = air_resistance
        self.coordinates = coordinates
    def update(self):
        self.agent.velocity.y += self.gravity
        self.agent.velocity.x *= self.air_resistance
        if abs(self.agent.velocity.x) <= 0.0001:
            self.agent.velocity.x = 0
        self.agent.position += self.agent.velocity
        if self.check_game_over():
            self.game_over()

    def check_game_over(self) -> bool:
        game_over = True
        if not self.agent.position.x + self.agent.size >= self.coordinates.x + self.field_size.x:
           if not self.agent.position.x <= self.coordinates.x:
               if not self.agent.position.y + self.agent.size >= self.coordinates.y + self.field_size.y:
                   if not self.agent.position.y <= self.coordinates.y:
                       return False
        return True
      
    def handle_input(self, input_event: pygame.event.Event):
        key = input_event.key
        if key == pygame.K_SPACE:
            self.agent.velocity.y += -2
        elif key == pygame.K_a:
            self.agent.velocity.x += -0.5
        elif key == pygame.K_d:
            self.agent.velocity.x += 0.5
    def draw(self, screen: pygame.Surface):
        pygame.draw.rect(screen, "green", (self.agent.position, self.field_size/10))
        pygame.draw.rect(screen, "white", (self.coordinates.x, self.coordinates.y, self.coordinates.x + self.field_size.x, self.coordinates.y + self.field_size.y), width=1)

    def game_over(self):
        self.agent = Agent(position=self.coordinates+(self.field_size/2), size=self.field_size.x/10, user_controlled=self.agent.user_controlled)

pygame.init()
screen = pygame.display.set_mode((800, 800))
pygame.display.set_caption("Genetic algorithm")
clock = pygame.time.Clock()

games: list[Game] = []

# game = Game(coordinates=pygame.Vector2((0, 0)), field_size=pygame.Vector2((800, 800)), user_controlled=True)
# games.append(game)

n_games = 64
for x in range(int(sqrt(n_games))):
    for y in range(int(sqrt(n_games))):
        field_wh = screen.get_size()[0]/int(sqrt(n_games))
        game = Game(coordinates=pygame.Vector2((x*100, y*100)), field_size=pygame.Vector2((field_wh, field_wh)), user_controlled=True)
        games.append(game)

running = True

while running:

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
            for game in games:
                if game.agent.user_controlled:
                    game.handle_input(event)

        
    screen.fill("black")
    for game in games:

        game.update()
        game.draw(screen)


    pygame.display.flip()
    clock.tick(60)

pygame.quit()