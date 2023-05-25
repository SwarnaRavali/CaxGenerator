import pyvista as pv
from typing import Any
import numpy as np
from pathlib import Path
from vcti.engine.multivaluedarray import MultiValuedArray
import vcti.cax.cax as caxlib
PATH_SEP = '/'
GEOM_NAME = 'Geometry'

ug = pv.UnstructuredGrid()
ug.cell_arrays()


class DataArray:
    def __init__(self, *args: Any, **kwds: Any) -> Any:
        pass


def child_path(parent_path, child_name):
    """
    returns concatenated string with PATH_SEP seperator
    """
    return f'{parent_path}{PATH_SEP}{child_name}'


class ResultData:
    def __init__(self, name, type, values, meshids, stepids) -> None:
        """
        Used to store the result data.

        Parameters:
        name: Result name
        type : ResultType
        values: Numpy array of result values.
        meshids: Mesh ids
        stepids : stepIds
        """
        self._initialize()
        self.name = name
        self.datasettype = type
        self.values = values
        self.mesh_ids = meshids
        self.step_ids = stepids

    def _initialize(self):
        self.name = None
        self.datasettype = None
        self.values = None
        self.mesh_ids = None
        self.step_ids = None


class CaxGenerator:
    def __init__(self,
                 cax_file: str or Path or caxlib.Model,
                 *args: Any, **kwds: Any
                 ) -> None:
        """
        Used to generate cax with data arrays.

        Parameters
        -----------
        cax_file:  CAx file path or caxlib Model

        Examples
        ---------\
        
        >>> from caxgenerator import CaxGenerator
        >>> cg = CaxGenerator(r"CaxFilePath.cax")
        >>> # add coordinates, polylengths, connectivity, partfacemva, partnames
        >>> cg.generate_cax()
        """
        self._initialize()
        if isinstance(cax_file, str):
            self._cax_file_path = cax_file
        elif isinstance(cax_file, Path):
            self._cax_file_path = str(cax_file)
        elif isinstance(cax_file, caxlib.Model):
            self._cax_model = cax_file

        if self._model_name is None:
            if self._cax_file_path is not None:
                self._model_name = Path(self._cax_file_path).stem
            else:
                self._model_name = "CAX-Model"

    def _initialize(self) -> None:
        """
        Initializes member variables
        """
        self._coordinates = None
        self._node_ids = None
        self._polylengths = None
        self._connectivity = None
        self._part_faces_mva = None
        self._part_names = None
        self._cax_file_path = None
        self._cax_model = None
        self._cae_model = None
        self._model_name = None
        self._step_list = None
        self._application_name = None
        self._model_type = "Geometry"
        self.mesh_ids = []
        self._result_list = None
        self._result_data = {}

    def set_caemodel(self, caemodel=None) -> None:
        """
        Set cae model from which the data has to be extracted.
        If None, the other data arrays can be used to export the data to cax
        """
        self._cae_model = caemodel

    def _get_paths(self, partnames) -> list:
        if isinstance(partnames, str):
            partnames = [partnames]
        model_path = child_path('', self.model_name)
        geometry_node_path = child_path(model_path, self.model_type)
        cax_paths = [child_path(geometry_node_path, x)
                     for x in partnames]
        return cax_paths

    def _get_cax_paths(self) -> list:
        if self.get_part_names() is None:
            raise Exception("Part names are empty.")
        cax_paths = self._get_paths(self._part_names)
        return cax_paths

    def add_node(self, nodetype: caxlib.NodeType, path: str, attrs: dict = None) -> int:
        geom_node_id = -1
        if self._cae_model is not None:
            geom_node_id = self._cax_model.add_node(nodetype, path, attrs)

        return geom_node_id

    def _add_paths(self) -> None:
        cax_paths = self._get_cax_paths()
        self.mesh_ids = self._cax_model.add_assembly_components(
            cax_paths, PATH_SEP)

    def _load_cax_model(self) -> None:
        if self._cax_model is not None:
            return

        if self._cax_file_path is None:
            raise RuntimeError("File path is not valid.")
        self._cax_model = caxlib.Model()

        self._cax_model.save_to(str(self._cax_file_path),
                                truncate_file=True,
                                application_name=self.get_applicationname(),
                                application_version='1.0',
                                change_history_comment='Test')

    def _add_step_data(self) -> None:
        if self._cax_model is None:
            raise Exception("CaxModel is not loaded.")

        if self.step_list is None:
            step_list = self.get_step_list()
            self.step_list = self.cax_model.create_step_list(step_list)
        elif isinstance(self.step_list, list):
            if isinstance(self.step_list[0], tuple):
                self.step_list = self.cax_model.create_step_list(
                    self.step_list)
        else:
            raise Exception("Improper step data.")

    def set_applicationname(self, applicationname) -> None:
        """
        Application name
        """
        self._application_name = applicationname

    def get_applicationname(self) -> str:
        """
        Default application name. Can be overridden by child class.
        """
        if self._application_name is None:
            self._application_name = "CaxGenerator"
        return self._application_name

    def set_modelname(self, model_name) -> None:
        """
        Model Name
        """
        self._model_name = model_name

    def set_model_type(self, model_type="Geometry") -> None:
        """
        set the Model type, default is Geometry
        """
        self._model_type = model_type

    def get_model_type(self) -> str:
        """
        Get the Model type. Default type is Geometry.
        """
        if self._model_type is None:
            self._model_type = "Geometry"

        return self._model_type

    def set_step_list(self, step_list) -> None:
        """
        Set step list. Can be list of tuple with Instance info,
        CaxLib stepData object.
        """
        self._step_list = step_list

    def get_step_list(self, step_list) -> list:
        """
        Get step list
        Default is L1
        """
        step_list = [("L1", {})]
        self._step_list = step_list
        return self._step_list

    def set_part_names(self, part_names) -> None:
        """
        Part Names
        """
        self._part_names = part_names

    def get_part_names(self) -> list:
        """
        Part Names
        """
        return self._part_names

    def set_coordinates(self, coordinates) -> None:
        """
        set Coordinates.
        Coordinates should be a numpy array.
        """
        self._coordinates = coordinates

    def get_coordinates(self) -> np.ndarray:
        """
        Get Coordinates. This function can be overridden by child class.
        """
        return self._coordinates

    def set_nodeids(self, nodeids) -> None:
        """
        Set NodeIds.
        Node Ids should be a numpy array.
        """
        self._node_ids = nodeids

    def get_nodeids(self) -> np.ndarray:
        """
        Get NodeIds. 
        This function can be overridden by child class.
        """
        return self._node_ids

    def set_polylengths(self, polylengths) -> None:
        """
        Set Polylengths.
        Node Ids should be a numpy array.
        """
        self._polylengths = polylengths

    def get_polylengths(self) -> np.ndarray:
        """
        Get Elemental Polylengths. 
        This function can be overridden by child class.
        """
        return self._polylengths

    def set_connectivity(self, connectivity) -> None:
        """
        Set connectivity.
        Node Ids should be a numpy array.
        """
        self._connectivity = connectivity

    def get_connectivity(self) -> np.ndarray:
        """
        Get Element connectivity. 
        This function can be overridden by child class.
        """
        return self._connectivity

    def set_result_list(self, result_list) -> None:
        """
        Set Results list.
        Node Ids should be a numpy array.
        """
        self._result_list = result_list

    def get_results_list(self) -> list:
        """
        Get List of results available. 
        This function can be overridden by child class.
        """
        return self._result_list

    def set_result_data(self, i, result_data) -> None:
        """
        Set Result data at index.
        Node Ids should be a numpy array.
        """
        self._result_data[i] = result_data

    def get_result_values(self, i) -> ResultData:
        """
        Get Result Data Object at index. 
        This function can be overridden by child class.
        """
        self._result_data.get(i, None)

    def set_part_face_mva(self, part_face_mva) -> None:
        """
        Set MultivaluedArray object of Parts and their faces data.
        """
        self._part_faces_mva = part_face_mva

    def create_part_face_mva(self) -> MultiValuedArray:
        """
        Function to create a Part_face_mva variable.
        This function has to be overridden by subclass.
        """
        return self._part_faces_mva

    def add_components(self, part) -> list:
        """
        Add Component to the cax model. where part is the name of the part.
        returns mesh id of the added component.
        """
        cax_path = self._get_paths(part)
        if self._cax_model is None:
            raise Exception("CaxModel is not loaded.")

        mesh_ids = self._cax_model.add_assembly_components(
            cax_path, PATH_SEP)

        for mid in mesh_ids:
            self.mesh_ids.append(mid)

        return mesh_ids

    def load_cax_model(self) -> None:
        """
        Load the cax model
        """
        try:
            self._load_cax_model()
        except:
            raise Exception("Failed to export cax.")

    def add_paths(self) -> None:
        """
        Add partnames to the cax model.
        Has to be called after loading the cax model function.
        """
        self._add_paths()

    def add_step_data(self) -> None:
        """
        Add stepdata to the cax model.
        Has to be called after loading the cax model function.
        """
        self._add_step_data()

    def add_array(self, datasettype, array, meshids, steplist) -> None:
        """
        Add array to the cax model.
        """
        self._cax_model.add_array_property(
            datasettype, array, meshids, steplist)

    def generate_cax(self) -> None:
        """
        Generates cax. Has to be called after adding all the data arrays 
        required to export to cax.
        This function can be overridden by sub classes
        """
        try:
            self._load_cax_model()
        except:
            raise Exception("Failed to export cax.")

        self._add_paths()
        self._add_step_data()

        self.write_geometry()
        self.add_results()

    def write_geometry(self) -> None:
        """
        Exports Geometry dato to cax
        """
        if self.get_part_names() is not None:
            if len(self._part_faces_mva.counts()) != len(self.get_part_names()):
                raise Exception("Part names doesnt match with Connectivity.")
        else:
            self._part_names = []
            for i in range(len(self._part_faces_mva.counts())):
                self._part_names.append(f"Part_{i+1}")

        self.create_part_face_mva()
        nb_parts = self._part_faces_mva.counts().shape
        for ind in range(nb_parts):
            poly_elem = self._part_faces_mva.values()[self._part_faces_mva.offsets(
            )[ind]: self._part_faces_mva.offsets()[ind+1]].astype(np.uint32)
            mesh_id = self.mesh_ids[ind]
            self.add_array(caxlib.DatasetType.POLYGON_ELEMENTS,
                           poly_elem, mesh_id, self._step_list)

        self.add_array(caxlib.DatasetType.POLYGON_CONNECTIVITY,
                       self.get_connectivity(), self.mesh_ids,
                       self._step_list)
        self.add_array(caxlib.DatasetType.POLYGON_LENGTHS,
                       self.get_polylengths(), self.mesh_ids,
                       self._step_list)
        self.add_array(caxlib.DatasetType.COORDINATES,
                       self.get_coordinates(), self.mesh_ids,
                       self._step_list)
        self.add_array(caxlib.DatasetType.NODE_IDS,
                       self.get_nodeids(), self.mesh_ids,
                       self._step_list)

    def add_results(self) -> None:
        """
        Adds result to cax model
        """
        result_list = self.get_results_list()

        for i, result in enumerate(result_list):
            result_data = self.get_result_values(i)
            self._cax_model.add_named_array_property(result_data.name, result_data.datasettype,
                                                     result_data.values, result_data.mesh_ids,
                                                     result_data.step_ids)

    def add_result_data(self, result_name, result_data, Datasettype, meshids, steplist) -> None:
        self._cax_model.add_named_array_property(result_name, Datasettype,
                                                 result_data, meshids,
                                                 steplist)

    def close(self) -> None:
        """
        Closes the cax model.
        """
        if self._cax_model is not None:
            self._cax_model.close()
