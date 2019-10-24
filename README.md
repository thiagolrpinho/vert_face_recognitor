# vert_face_recognitor
## How
The algorithm first convert the images in the database to an embbeded code that 
can be used to compare faces similarities
Then compares a new input with the know database and verifies if its one of them
### Enconding Database
To run it: 
```
python encode_database.py --database known_people
```
Where database is the name of the folder in the same directory with the know people database.
## Finding Person
To run it:
```
python recognize_face.py --image test_images/CNH.jpg
```
Where image is the relative path to the image you want to find a person

