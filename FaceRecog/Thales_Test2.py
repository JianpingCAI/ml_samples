import os
import numpy as np

# dummy function
# def img_to_encoding(file_path, model):
#     return file_path


def distance_L2(enc1, enc2):
    """
    Compute the L2 distance between two encodings.
    Suppose encoding is a numpy array.
    """
    # return np.linalg.norm(enc1-enc2)
    # L2 implementation
    diff = enc1 - enc2
    return np.sqrt((diff*diff).sum())


def enrolDB(model):
    """
    people to enroll: A, B, C, D, E, F, G, H
    """
    persons = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    database = {}
    path = 'images'
    for p in persons:
        database[p] = img_to_encoding(os.path.join(path, p + '.png'), model)

    return database


def who_is_it(image_path, database, model):
    """
    Detect the name of the input 'image_path'
    """
    # Compute the target encoding of the image from image_path
    image_enc = img_to_encoding(image_path, model)

    # Find the encoding from the database that has smallest distance with the target encoding.
    # Loop over the database dictionary's names and encodings and Compute the L2 distance between the target "encoding" and the current
    # "encoding" from the database.
    min_dist = np.finfo(float).max
    id = None
    for (db_name, db_enc) in database.items():
        dist = distance_L2(image_enc, db_enc)

        if(dist < min_dist):
            min_dist = dist
            id = db_name

    # The method is to return the smallest distance and identity of the person
    print('Name: {}'.format(id))
    return id


if __name__ == '__main__':
    #FRmodel = {}
    database = enrolDB(FRmodel)
    who_is_it("images/camera_0.jpg", database, FRmodel)
