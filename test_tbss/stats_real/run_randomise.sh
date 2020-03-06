for modality in FA 
#for modality in FA FAt FW
do
    /data/pnl/kcho/PNLBWH/fsl_randomise/randomise_parallel_pnl \
        -i all_${modality}_skeletonized.nii.gz \
        -o tbss_${modality} \
        -d design.mat \
        -t design.con \
        -m ENIGMA_DTI_FA_skeleton_mask.nii.gz \
        -n 5000 \
        --T2
done
