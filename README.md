In order to generate 3d meshs based on 2d images first you have to create a new conda environement using those commands
```bash
conda create --name img2mesh python=3.9.20
conda activate img2mesh
```
Then you have to setup the CUDA environement:
```bash
pip install torch==2.5.0 torchvision --index-url https://download.pytorch.org/whl/test/cu121
```
then install all requirements:
```bash
pip install numpy==1.26.4
pip install opencv-python==4.10.0.84
pip install pyopengl==3.1.0
pip install scikit-image==0.24.0
pip install scipy
pip install tensorboard
pip install smplx==0.1.28
pip install spacepy==0.6.0
pip install torchgeometry==0.1.2
pip install tqdm
pip install trimesh==4.5.1
pip install pyrender==0.1.45
pip install git+https://github.com/mattloper/chumpy.git
```
or 
```bash
pip install -r requirements.txt
```
to generate a 3d mesh run this command after inserting the image inside inputs folder:
```bash
python3 demo.py --checkpoint=data/model_checkpoint.pt --img=examples/im1010.jpg
```
the results is in obj format and can be found in output folder 🎉😎