'''
初始化及循环开始
'''
import pygame,sys,random
import flappy_bird_utils

SCREENWIDTH  = 288
SCREENHEIGHT = 512


BASEY = SCREENHEIGHT * 0.79


# 用于变更开局鸟的位置
# PLAYER_INDEX_GEN = cycle([0, 1, 2, 1])

pygame.init()

screen=pygame.display.set_mode((SCREENWIDTH,SCREENHEIGHT))


IMAGES, SOUNDS, HITMASKS = flappy_bird_utils.load()
PLAYER_WIDTH = IMAGES['player'][0].get_width()
PLAYER_HEIGHT = IMAGES['player'][0].get_height()
PIPE_WIDTH = IMAGES['pipe'][0].get_width()
PIPE_HEIGHT = IMAGES['pipe'][0].get_height()
BACKGROUND_WIDTH = IMAGES['background'].get_width()

pygame.display.set_caption("Hello Pygame!")


PIPEGAPSIZE = 100
gapYs = [20, 30, 40, 50, 60, 70, 80, 90]
index = random.randint(0, len(gapYs) - 1)
gapY = gapYs[index]
gapY += int(BASEY * 0.2)
pipeX = SCREENWIDTH - 130
newPipe = [
        {'x': pipeX, 'y': gapY - PIPE_HEIGHT},  # upper pipe
        {'x': pipeX, 'y': gapY + PIPEGAPSIZE},  # lower pipe
    ]

print(newPipe)
print(PLAYER_HEIGHT)
print(PIPE_WIDTH)


screen.blit(IMAGES['pipe'][0], (newPipe[0]['x'], newPipe[0]['y']))
screen.blit(IMAGES['pipe'][1], (newPipe[1]['x'], newPipe[1]['y']))

screen.blit(IMAGES['pipe'][0], (newPipe[0]['x']+PIPE_WIDTH, newPipe[0]['y']+ PIPE_HEIGHT+50  ))
pygame.display.update()
'''
事件接收及响应
'''
while True:
    for event in pygame.event.get():
        if event.type==pygame.QUIT:
            sys.exit()
    pygame.display.update()













#   控制台输出带颜色的代码示例
# class bcolors:
#     HEADER = '\033[95m'
#     OKBLUE = '\033[94m'
#     OKGREEN = '\033[92m'
#     WARNING = '\033[93m'
#     FAIL = '\033[91m'
#     ENDC = '\033[0m'
#     BOLD = '\033[1m'
#     UNDERLINE = '\033[4m'
#
# print(bcolors.FAIL + "Warning: xxxx"+bcolors.ENDC)
#
#


