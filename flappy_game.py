import gymnasium as gym
import flappy_bird_gymnasium
import pygame

# Create environment
env = gym.make("FlappyBird-v0", render_mode="human")

state, info = env.reset()

done = False

while not done:
    action = 0

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True

        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                action = 1  # Flap

    next_state, reward, terminated, truncated, info = env.step(action)

    done = terminated or truncated

    env.render()

env.close()
pygame.quit()