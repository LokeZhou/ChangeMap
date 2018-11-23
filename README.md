# ChangeMap

Usage

Dependencies

Python2.7
PyTorch(0.4.0+)
torchvision 0.2.0 (higher version may cause issues)



Train
As an example, use the following command to train a PSMNet on Your Dataset:

python main.py --maxdisp 192 \
               --model stackhourglass \
               --datapath (your data folder)\
               --epochs 10 \
               —-enablecuda True \ 
               --savemodel (path for saving model)

Tip:
1.Make sure your dataset file construction as :
  —-your data folder
     —-trian
        —-gt
            —-gt1.png(.tif……),….
        —-left
            —-left_trian_im1.png(.tif……),……
        —-right
            —-right_trian_im1.png(.tif……),……
      —-test
        —-gt
           —-gt1.png(.tif……),….
        —-left
            —-left_test_im1.png(.tif……),……
        —-right
            —-right_test_im1.png(.tif……),……

2.if no cuda ,set —-enablecuda as False

Evaluation
Use the following command to evaluate the trained PSMNet on Your test data:

python submission.py --maxdisp 192 \
                     --model stackhourglass \
                     --enablecuda True \
                     --datapath (your test data folder) \
                     --loadmodel (finetuned PSMNet) \


Tip:
1.Make sure your dataset file construction as :
  —-your data folder
      —-test
        —-left
          —-left_test_im1.png(.tif……),……
        —-right
          —-right_test_im1.png(.tif……),……

 2.if no cuda ,set —-enablecuda as False

