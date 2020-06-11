
class RandomiseRun(RandomiseConMat):
    """Randomise output class

    This class is used to contain information about FSL randomise run,
    from a randomise stats output directory. The input directory should
    have following files inside it.

    1. randomise output statistic files. (one more more)
        - `*${modality}*corrp*.nii.gz`
        - make sure the modality is included in the stat file name.
    2. merged skeleton file for all subjects.
        - `all_${modality}*_skeleton.nii.gz`
        - make sure the modality is included in the merged skeleton file name.
    3. design matrix and design contrast files used for the randomise.
        - It will be most handy to have them named `design.con` and
          `design.mat`

    Key arguments:
        location: str or Path object of a randomise output location.
                  Preferably with a 'design.con' and 'design.mat' in the same
                  directory.
        contrast_file: str, randomise contrast file.
                       default='design.con'
        matrix_file: str, randomise matrix file.
                     default='design.mat'
    """
    def __init__(self,
                 location:str='.',
                 contrast_file:Union[bool, str]=False,
                 matrix_file:Union[bool, str]=False):
        # define inputs
        self.location = Path(location)

