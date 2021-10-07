#!/bin/sh
export FSLOUTPUTTYPE='NIFTI'

echo "The base name is $6"
echo "processing file from $1: $2"

###SECTION 1: SETUP YOUR DIRECTORY STRUCTURE FOR OPERATIONS WITHIN (ROOT)/TMP ###

#go into MRI folders
cd $1
#copy a subject to the temporary directory within the overall directory
cp $2 $3
#head over to the temporary directory
cd $3
#rename your named file to T1.nii.gz
mv $2 T1.nii.gz


###SECTION 2: PIPELINE UP UNTI BIAS FIELD CORRECTION WITH FAST###

#step1 is to swap axes so that the brain is in the same direction as MNI template.
${FSLDIR}/bin/fslreorient2std T1.nii.gz T1_reoriented.nii

#step2 is to estimate robust field of view
line=`${FSLDIR}/bin/robustfov -i T1_reoriented.nii | grep -v Final | head -n 1`

x1=`echo ${line} | awk '{print $1}'`
x2=`echo ${line} | awk '{print $2}'`
y1=`echo ${line} | awk '{print $3}'`
y2=`echo ${line} | awk '{print $4}'`
z1=`echo ${line} | awk '{print $5}'`
z2=`echo ${line} | awk '{print $6}'`

x1=`printf "%.0f", $x1`
x2=`printf "%.0f", $x2`
y1=`printf "%.0f", $y1`
y2=`printf "%.0f", $y2`
z1=`printf "%.0f", $z1`
z2=`printf "%.0f", $z2`

#step3 is to cut the brain to get area of interest (roi), sometimes it cuts part of the brain
${FSLDIR}/bin/fslmaths T1_reoriented.nii -roi $x1 $x2 $y1 $y2 $z1 $z2 0 1 T1_roi.nii

#step4: remove skull -g 0.1 -f 0.45
${FSLDIR}/bin/bet T1_roi.nii T1_brain.nii -R 

#step5: registration from cut to MNI
${FSLDIR}/bin/flirt -in T1_brain.nii -ref $FSLDIR/data/standard/MNI152_T1_1mm_brain -omat orig_to_MNI.mat

#step6: apply matrix onto reoriented original image
${FSLDIR}/bin/flirt -in T1_reoriented.nii -ref $FSLDIR/data/standard/MNI152_T1_1mm_brain -applyxfm -init orig_to_MNI.mat -out T1_MNI.nii

#step7: skull remove -f 0.3 -g -0.0
${FSLDIR}/bin/bet T1_MNI.nii T1_MNI_brain.nii -R 

# step8: register the skull removed scan to MNI_brain_only template again to fine tune the alignment
${FSLDIR}/bin/flirt -in T1_MNI_brain.nii -ref $FSLDIR/data/standard/MNI152_T1_1mm_brain -out T1_MNI_brain.nii


###SECTION 3: FAST for bias field correction (added bonus: gray matter + white matter segmentation)
#-B = output bias-corrected image
${FSLDIR}/bin/fast -B T1_MNI_brain.nii

###SECTION 4: FNIRT FOR FINAL NON-LINEAR REGISTRATION OF MNI ATALS TO BIAS-FIELD-CORRECTED IMAGE ###

#derive map from input image to
#${FSLDIR}/bin/fnirt --in=T1_MNI_brain --cout=map --config=/home/mattmill/fnirt_config

#fslmaths for twofold subsampling
#fslmaths T1_MNI_brain -subsamp2 T1_2mm

#invwarp --ref=T1_2mm --warp=map --out=inv_map

#applywarp --ref=T1_MNI_brain --in=/home/mattmill/MNI_atlas --warp=inv_map --out=$5/segmented_$2 --interp=nn

#move registered T1 brain to separate registry in order to check registration
mv T1_MNI_brain.nii T1_MNI_brain_$6.nii
mv T1_MNI_brain_$6.nii $4

#clean up tmp directory
rm -f $3/*

