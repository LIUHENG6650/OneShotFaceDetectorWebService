# One Shot Face Detector Web Service
OSFD Web Service is flask microservice which does face detection and learns only with single picture. It uses approximate near neighbour search based on euclidean distance and finds best fit face element based on 512 facenet feature vector. It uses sqlite as content db and annoy library for near neighbour indexing. Facenet is implemented on tensorflow 1.13.1.
### Requirements

 - Python 3.5 and above
 - pip
 - Windows, Linux or MacOS(Not Tested)

### Installation
 - Clone the repository

	    git clone https://github.com/mehmetozturk4705/OneShotFaceDetectorWebService
 - Download models weights from [here](https://drive.google.com/file/d/1kpprV7bAgkEZ-FlmZoZ0zMxgUKnZTBmv/view?usp=sharing)
 - Extract files and move **models** folder near the **app.py**
 - Install requirements (Before installation I recommend you to create virtual environment)


	    pip install -r requirements.txt
	    flask run

### How to use web service
OSFD uses annoy backend to utilize near neigbour search. After adding faces it needs to index face features.
#### Add Face
Adds new face with **image** file of **name**.  If image has multiple faces, it selects biggest one.

	localhost:5000/add   	[POST]

	 - image ->> png or jpeg image file of face
	 - name ->> name of face
#### Delete Id
Removes all faces of **name**

	localhost:5000/delete   [POST]

	 - name ->> name of face
#### Balance
After adding or deleting face you should always call balance in order to take effect.

	localhost:5000/balance  [POST]

#### Detect Face
Detects face in **image** file. If image has multiple faces, it selects biggest one.

	localhost:5000/detect   [POST]

	 - image ->> png or jpeg image file of face
	 - threshold [optional] ->> Euclidean threshold
Returns:

	{
    "data": {
        "distance": 0.25,
        "id": "Face 1"
	    },
    "success": true
	}

### Tuning
OSFD uses euclidean threshold in order to decide whether input features are same person. You can tune threshold in **configuration.json** or threshold field of **detect** endpoint.
Higher threshold means system is much more open to confuse faces. Lower threshold means harder to detect.

### Reference

 - [https://github.com/spotify/annoy](https://github.com/spotify/annoy) for annoy indexing.
 - [https://github.com/davidsandberg/facenet](https://github.com/davidsandberg/facenet) for tensorflow implementation of **Facenet**

### For Information
Please send email to [bilgi@pyturk.com](mailto:bilgi@pyturk.com)
