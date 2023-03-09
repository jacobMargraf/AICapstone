import PySimpleGUI as pg

global gui_active
gui_active = True

def run_dino():
    global gui_active
    ######################################### DINO JUMP ##################################################################
    import pygame
    import os
    import random
    import sys
    import math
    import neat

    import gzip
    import pickle
    import time

    from neat.population import Population
    from neat.reporting import BaseReporter

    pygame.init()
    # Global Constants
    SCREEN_HEIGHT = 600
    SCREEN_WIDTH = 1100
    SCREEN = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

    RUNNING = [pygame.image.load(os.path.join("Assets/Dino", "DinoRun1.png")),
               pygame.image.load(os.path.join("Assets/Dino", "DinoRun2.png"))]

    JUMPING = pygame.image.load(os.path.join("Assets/Dino", "DinoJump.png"))

    SMALL_CACTUS = [pygame.image.load(os.path.join("Assets/Cactus", "SmallCactus1.png")),
                    pygame.image.load(os.path.join("Assets/Cactus", "SmallCactus2.png")),
                    pygame.image.load(os.path.join("Assets/Cactus", "SmallCactus3.png"))]
    LARGE_CACTUS = [pygame.image.load(os.path.join("Assets/Cactus", "LargeCactus1.png")),
                    pygame.image.load(os.path.join("Assets/Cactus", "LargeCactus2.png")),
                    pygame.image.load(os.path.join("Assets/Cactus", "LargeCactus3.png"))]

    BG = pygame.image.load(os.path.join("Assets/Other", "Track.png"))

    FONT = pygame.font.Font('freesansbold.ttf', 20)

    class Dinosaur:
        X_POS = 80
        Y_POS = 310
        JUMP_VEL = 8.5

        def __init__(self, img=RUNNING[0]):
            self.image = img
            self.dino_run = True
            self.dino_jump = False
            self.jump_vel = self.JUMP_VEL
            self.rect = pygame.Rect(self.X_POS, self.Y_POS, img.get_width(), img.get_height())
            self.color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            self.step_index = 0

        # Check if the dinosaur is running or jumping
        def update(self):
            if self.dino_run:
                self.run()
            if self.dino_jump:
                self.jump()
            # Animation stuff
            if self.step_index >= 10:
                self.step_index = 0

        def jump(self):
            self.image = JUMPING
            if self.dino_jump:
                self.rect.y -= self.jump_vel * 4
                self.jump_vel -= 0.8
            if self.jump_vel <= -self.JUMP_VEL:
                self.dino_jump = False
                self.dino_run = True
                self.jump_vel = self.JUMP_VEL

        # changes image of the dinosaur
        def run(self):
            self.image = RUNNING[self.step_index // 5]
            self.rect.x = self.X_POS
            self.rect.y = self.Y_POS
            self.step_index += 1

        # display image onto screen
        def draw(self, SCREEN):
            SCREEN.blit(self.image, (self.rect.x, self.rect.y))
            pygame.draw.rect(SCREEN, self.color, (self.rect.x, self.rect.y, self.rect.width, self.rect.height), 2)
            for obstacle in obstacles:
                pygame.draw.line(SCREEN, self.color, (self.rect.x + 54, self.rect.y + 12), obstacle.rect.center, 2)

    class Obstacle:
        def __init__(self, image, number_of_cacti):
            self.image = image
            self.type = number_of_cacti
            self.rect = self.image[self.type].get_rect()
            self.rect.x = SCREEN_WIDTH

        def update(self):
            self.rect.x -= game_speed
            if self.rect.x < -self.rect.width:
                obstacles.pop()

        def draw(self, SCREEN):
            SCREEN.blit(self.image[self.type], self.rect)

    class SmallCactus(Obstacle):
        def __init__(self, image, number_of_cacti):
            super().__init__(image, number_of_cacti)
            self.rect.y = 325

    class LargeCactus(Obstacle):
        def __init__(self, image, number_of_cacti):
            super().__init__(image, number_of_cacti)
            self.rect.y = 300

    class Checkpointer(BaseReporter):
        """
        A reporter class that performs checkpointing using `pickle`
        to save and restore populations (and other aspects of the simulation state).
        """

        def __init__(self, generation_interval=100, time_interval_seconds=300,
                     filename_prefix='neat-checkpoint-'):
            """
            Saves the current state (at the end of a generation) every ``generation_interval`` generations or
            ``time_interval_seconds``, whichever happens first.

            :param generation_interval: If not None, maximum number of generations between save intervals
            :type generation_interval: int or None
            :param time_interval_seconds: If not None, maximum number of seconds between checkpoint attempts
            :type time_interval_seconds: float or None
            :param str filename_prefix: Prefix for the filename (the end will be the generation number)
            """
            self.generation_interval = generation_interval
            self.time_interval_seconds = time_interval_seconds
            self.filename_prefix = filename_prefix

            self.current_generation = None
            self.last_generation_checkpoint = -1
            self.last_time_checkpoint = time.time()

        def start_generation(self, generation):
            self.current_generation = generation

        def end_generation(self, config, population, species_set):
            checkpoint_due = False

            if self.time_interval_seconds is not None:
                dt = time.time() - self.last_time_checkpoint
                if dt >= self.time_interval_seconds:
                    checkpoint_due = True

            if (checkpoint_due is False) and (self.generation_interval is not None):
                dg = self.current_generation - self.last_generation_checkpoint
                if dg >= self.generation_interval:
                    checkpoint_due = True

            if checkpoint_due:
                self.save_checkpoint(config, population, species_set, self.current_generation)
                self.last_generation_checkpoint = self.current_generation
                self.last_time_checkpoint = time.time()

        def save_checkpoint(self, config, population, species_set, generation):
            """ Save the current simulation state. """
            filename = '{0}{1}'.format(self.filename_prefix, generation)
            print("Saving checkpoint to {0}".format(filename))

            with gzip.open(filename, 'w', compresslevel=5) as f:
                data = (generation, config, population, species_set, random.getstate())
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

        @staticmethod
        def restore_checkpoint(filename):
            """Resumes the simulation from a previous saved point."""
            with gzip.open(filename) as f:
                generation, config, population, species_set, rndstate = pickle.load(f)
                random.setstate(rndstate)
                return Population(config, (population, species_set, generation))


    def remove(index):
        dinosaurs.pop(index)
        ge.pop(index)
        nets.pop(index)

    def distance(pos_a, pos_b):
        dx = pos_a[0] - pos_b[0]
        dy = pos_a[1] - pos_b[1]
        return math.sqrt(dx ** 2 + dy ** 2)

    def eval_genomes(genomes, config):
        global game_speed, x_pos_bg, y_pos_bg, points, obstacles, dinosaurs, ge, nets
        clock = pygame.time.Clock()
        points = 0

        obstacles = []
        dinosaurs = []
        ge = []
        nets = []

        x_pos_bg = 0
        y_pos_bg = 380
        game_speed = 20

        for genome_id, genome in genomes:
            dinosaurs.append(Dinosaur())
            ge.append(genome)
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            nets.append(net)
            genome.fitness = 0

        def score():
            global points, game_speed
            points += 1
            if points % 100 == 0:
                game_speed += 1
            text = FONT.render(f'Points: {str(points)}', True, (0, 0, 0))
            SCREEN.blit(text, (950, 50))

        def statistics():
            global dinosaurs, game_speed, ge
            text_1 = FONT.render(f'Dinosaurs Alive:  {str(len(dinosaurs))}', True, (0, 0, 0))
            text_2 = FONT.render(f'Generation:  {pop.generation + 1}', True, (0, 0, 0))
            text_3 = FONT.render(f'Game Speed:  {str(game_speed)}', True, (0, 0, 0))

            SCREEN.blit(text_1, (50, 450))
            SCREEN.blit(text_2, (50, 480))
            SCREEN.blit(text_3, (50, 510))

        def background():
            global x_pos_bg, y_pos_bg
            image_width = BG.get_width()
            SCREEN.blit(BG, (x_pos_bg, y_pos_bg))
            SCREEN.blit(BG, (image_width + x_pos_bg, y_pos_bg))
            if x_pos_bg <= -image_width:
                x_pos_bg = 0
            x_pos_bg -= game_speed

        run = True
        while run:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
            SCREEN.fill((255, 255, 255))

            for dinosaur in dinosaurs:
                dinosaur.update()
                dinosaur.draw(SCREEN)

            if len(dinosaurs) == 0:
                save_file = open("save_file.txt", "a")
                lines = []
                lines.append("\nGeneration: " + str(pop.generation+1))
                lines.append("High Score: " + str(points) + "\n")
                lines_str = "\n".join(lines)
                save_file.writelines(lines_str)
                save_file.close()
                break

            if len(obstacles) == 0:
                rand_int = random.randint(0, 1)
                if rand_int == 0:
                    obstacles.append(SmallCactus(SMALL_CACTUS, random.randint(0, 2)))
                elif rand_int == 1:
                    obstacles.append(LargeCactus(LARGE_CACTUS, random.randint(0, 2)))

            for obstacle in obstacles:
                obstacle.draw(SCREEN)
                obstacle.update()
                for i, dinosaur in enumerate(dinosaurs):
                    if dinosaur.rect.colliderect(obstacle.rect):
                        ge[i].fitness -= 1
                        remove(i)

            for i, dinosaur in enumerate(dinosaurs):
                output = nets[i].activate((dinosaur.rect.y,
                                           distance((dinosaur.rect.x, dinosaur.rect.y),
                                                    obstacle.rect.midtop)))
                if output[0] > 0.5 and dinosaur.rect.y == dinosaur.Y_POS:
                    dinosaur.dino_jump = True
                    dinosaur.dino_run = False

            statistics()
            score()
            background()
            clock.tick(30)
            pygame.display.update()

    # Setup the NEAT
    def run(config_path):
        global pop
        config = neat.config.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            config_path
        )

        pop = neat.Population(config)
        pop.run(eval_genomes, 50)

    if __name__ == '__main__':
        local_dir = os.path.dirname(__file__)
        config_path = os.path.join(local_dir, 'config.txt')
        run(config_path)

    gui_active = False
    ####################################################################################################################################




#step 1: Set theme
pg.theme("DarkAmber")

#Step 2: Create Layout
layout = [
    [pg.Button("Dino Jump")],
    [pg.Button("Snake"),
     pg.Button("Cancel")]
]

#Step 3: Create Window
window = pg.Window("Game Hub", layout)

#Step 4: Event Loop
while gui_active == True:
    event, values = window.read()
    if event == "Cancel" or event == pg.WIN_CLOSED:
        gui_active = False
    if event == "Dino Jump":
        gui_active = False
        run_dino()




#Step 5: Close Window
window.close()