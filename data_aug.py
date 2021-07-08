from generator_methods import generator
import imageio
import imgaug as ia
import time

NUM_OF_HOME_DATA = 10
NUM_OF_SETTINGS_DATA = 11
NUM_OF_SEARCH_DATA = 7
NUM_OF_VERT_MORE_DATA = 6
NUM_OF_MENU_DATA = 6

path = 'data/'

path_result = 'data/'

index = 1
index_output = 1


def process_label(label, maxData):
    global index
    global index_output

    index = 1
    while index <= maxData:
        images = generator(path + label + '/' + label + str(index) + '.png')
        for i in range(len(images)):
            imageio.imwrite(path_result + label + str(index_output) + '.png', images[i])
            index_output += 1
        index += 1


# Generating icon
process_label('home', NUM_OF_HOME_DATA)
print("Complete home data !!!")
process_label('settings', NUM_OF_SETTINGS_DATA)
print("Complete settings data !!!")
process_label('search', NUM_OF_SEARCH_DATA)
print("Complete search data !!!")
process_label('vert_more', NUM_OF_VERT_MORE_DATA)
print("Complete vertical more data !!!")
process_label('menu', NUM_OF_MENU_DATA)
print("Complete menu data !!!")

# generator('darknet/data/home/home1.png')


