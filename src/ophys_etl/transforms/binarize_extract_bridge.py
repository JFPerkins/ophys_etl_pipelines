import argschema
import json
from pathlib import Path
from marshmallow import ValidationError

from ophys_etl.schemas import DenseROISchema, ExtractROISchema
from ophys_etl.extractors.motion_correction import MotionBorder
from ophys_etl.transforms.roi_transforms import dense_to_extract


class BridgeInputSchema(argschema.ArgSchema):
    input_file = argschema.fields.InputFile(
        required=True,
        description=("an output from BinarizerAndROICreator in "
                     "convert_rois.py"))
    storage_directory = argschema.fields.OutputDir(
        required=True,
        description="the intended destination directory for the traces")
    # NOTE these schema field names match convert_rois and not AllenSDK
    motion_corrected_video = argschema.fields.Str(
        required=True,
        validate=lambda x: Path(x).exists(),
        description=("Path to motion corrected video file *.h5"))
    motion_correction_values = argschema.fields.InputFile(
        required=True,
        description=("Path to motion correction values for each frame "
                     "stored in .csv format. This .csv file is expected to"
                     "have a header row of either:\n"
                     "['framenumber','x','y','correlation','kalman_x',"
                     "'kalman_y']\n['framenumber','x','y','correlation',"
                     "'input_x','input_y','kalman_x',"
                     "'kalman_y','algorithm','type']"))


class MotionBorderSchema(argschema.schemas.DefaultSchema):
    x0 = argschema.fields.Float()
    x1 = argschema.fields.Float()
    y0 = argschema.fields.Float()
    y1 = argschema.fields.Float()


class BridgeOutputSchema(argschema.schemas.DefaultSchema):
    # NOTE these schema field names match AllenSDK and not convert_rois
    motion_border = argschema.fields.Nested(
            MotionBorderSchema,
            required=True)
    storage_directory = argschema.fields.OutputDir(
            required=True)
    motion_corrected_stack = argschema.fields.Str(
            validate=lambda x: Path(x).exists(),
            required=True)
    rois = argschema.fields.Nested(
            ExtractROISchema,
            required=True,
            many=True)
    log_0 = argschema.fields.InputFile(
            required=True)


class BinarizeToExtractBridge(argschema.ArgSchemaParser):
    """
    This module can bridge the output of
    'ophys_etl_pipelines/src/ophys_etl/transforms/convert_rois.py'
    to the input of
    https://github.com/AllenInstitute/AllenSDK/blob/7e60bc5a811f76750d22a507f449621a0784e6bd/allensdk/brain_observatory/ophys/trace_extraction/_schemas.py#L33-L41  # noqa
    In production, this purpose is served by a LIMS strategy. This python
    bridge is here as a helper for running the pipeline manually outside
    of production
    """
    default_schema = BridgeInputSchema
    default_output_schema = BridgeOutputSchema

    def run(self):
        self.logger.name = type(self).__name__

        with open(self.args['input_file'], "r") as f:
            compatible_rois = json.load(f)

        # validate ROIs
        errors = DenseROISchema(many=True).validate(compatible_rois)
        if any(errors):
            raise ValidationError(f"Schema validation errors: {errors}")

        # read the motion border and check they are all the same
        for i, roi in enumerate(compatible_rois):
            iborder = MotionBorder(
                up=roi['max_correction_up'],
                down=roi['max_correction_down'],
                left=roi['max_correction_left'],
                right=roi['max_correction_right'])
            if i == 0:
                border = iborder
            else:
                assert iborder == border

        border_dict = {
                'y1': border.up,
                'y0': border.down,
                'x0': border.right,
                'x1': border.left
                }

        converted_rois = [dense_to_extract(roi) for roi in compatible_rois]

        output = {
                'log_0': self.args['motion_correction_values'],
                'motion_corrected_stack': self.args['motion_corrected_video'],
                'storage_directory': self.args['storage_directory'],
                'rois': converted_rois,
                'motion_border': border_dict
                }

        self.output(output, indent=2)
        self.logger.info(f"transformed {self.args['input_file']} to "
                         f"{self.args['output_json']}")


if __name__ == "__main__":  # pragma: nocover
    bridge = BinarizeToExtractBridge()
    bridge.run()
