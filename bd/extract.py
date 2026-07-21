
def extract_embedding(image, app):

    faces = app.get(image)

    if len(faces) == 0:
        return None

    face = faces[0]

    return face.embedding