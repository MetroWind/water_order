#!/usr/bin/env python3
# -*- coding: utf-8; -*-

# Copyright 2018 MetroWind <chris.corsair@gmail.com>
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# this program. If not, see <http://www.gnu.org/licenses/>.

from __future__ import print_function

import sys, os
import logging
import functools
import copy
import time
import multiprocessing as Mp
import multiprocessing.pool
import signal

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                "molmanip"))
import molecule
import linepy.matrix as Matrix

def getLogger(name="Order", level=logging.DEBUG):
    import sys
    Logger = logging.getLogger(name)
    Logger.setLevel(level)
    Handler = logging.StreamHandler(sys.stderr)
    Format = logging.Formatter("[%(levelname)s %(asctime)s] %(message)s",
                                  "%Y-%m-%d %H:%M:%S")
    Handler.setFormatter(Format)
    Logger.addHandler(Handler)
    return Logger

Logger = getLogger(level=logging.INFO)

def findNearest(mole, loc, n, boundary,
                near_limit=0.0001, atom_filter=lambda x: True):
    """For all the atoms in molecule `mole`, find the `n` nearest atoms to location
    `loc`, with boundary condition `boundary`. Atoms whose less than
    `near_limit` away are ignored. Only atoms that passes `atom_filter` are
    taken into accound.

    :type mole: molecule.Molecule
    :type loc: Vector3D
    :type n: int
    :type boundary: molecule.Boundary
    :rtype: molecule.Atom
    """

    Atoms = [atom for atom in mole if atom_filter(atom)]
    Atoms.sort(key=lambda at: boundary.dist(at.Loc, loc))

    Count = 0
    for Atom in Atoms:
        if boundary.dist(Atom.Loc, loc) < near_limit:
            continue
        elif Count < n:
            yield Atom
            Count += 1
        else:
            raise StopIteration

class Gridifier(object):
    def __init__(self, bound, grid_size):
        """
        :type bound: molecule.Boundary
        :type grid_size: int
        """
        self.Boundary = bound
        self._CellSize = 0.0

        # This is the number of cells in each direction, not the size of the
        # cells.
        self._GridSize = grid_size

        self._Points = []
        self._CellCount = 0
        self._init()

    def _init(self):
        # self._GridSize = tuple(int(self.Boundary.spanOfDim(i) / self._CellSize)
        #                        for i in range(self.Boundary.Dims))
        self._CellSize = self.Boundary.spanOfDim(0) / self._GridSize[0]
        self._CellCount = functools.reduce(lambda x, y: x*y, self._GridSize)
        self._Points = tuple(set() for i in range(self._CellCount))

    def gridCoord2GridIdx(self, coord):
        """Convert n-dimentional grid coordinate `coord` to 1D grid index."""
        NextDimSize = self._CellCount
        Idx = 0
        for i in range(self.Boundary.Dims):
            NextDimSize //= self._GridSize[i]
            Idx += coord[i] * NextDimSize

        return Idx

    def pointsAndDataAtGrid(self, coord):
        """Return a set of vectors and data assigned to grid coordinate `coord`. Each
        element in the returned set is a tuple containing the location vector
        and the data.
        """
        return self._Points[self.gridCoord2GridIdx(coord)]

    def gridIdx2GridCoord(self, idx):
        """Convert 1D grid index `idx` to n-dimentional grid coordinate."""

        NextDimSize = self._CellCount
        Coord = []
        for i in range(self.Boundary.Dims):
            NextDimSize //= self._GridSize[i]
            Coord.append(idx // NextDimSize)
            idx %= NextDimSize
        return tuple(Coord)

    def vec2GridCoord(self, v):
        """Assign grid coordinate to point `p`. Return self.

        :type v: Matrix.Vector
        :rtype: tuple
        """
        GridCoord = []
        for i in range(self.Boundary.Dims):
            GridPos = int((v[i] - self.Boundary.Walls[i][0]) / self._CellSize)
            if GridPos == self._GridSize[i]:
                GridPos = 0
            GridCoord.append(GridPos)
        Coord = self.senitizeGridCoord(GridCoord)
        return Coord

    def addPoint(self, p, data):
        """Assign grid coordinate to point `p`. And associate the point with `data`.
        Later you can get back all the associated data with the any grid
        coordinate. Argument `data` must be hashable.

        Return self.

        :type p: Matrix.Vector
        :type data: Any
        :rtype: Gridifier
        """
        GridCoord = self.vec2GridCoord(p)
        self._Points[self.gridCoord2GridIdx(GridCoord)].add((p, data))
        Logger.debug("{} assigned to {}.".format(p, GridCoord))

    def gridCoordShiftInDim(self, coord, i, delta):
        """Shift grid coordinate `coord` in `i`th direction by `delta`, and return the
        shifted coordinate.

        :type coord: tuple
        :type i: int
        :type delta: int
        :rtype: tuple
        """
        GridSize = self._GridSize[i]
        NewCoord = list(coord)
        NewCoord[i] = (NewCoord[i] + delta) % GridSize
        return tuple(NewCoord)

    def senitizeGridCoord(self, coord):
        """If grid coordinate `coord` is out of the box, shift it back into the box with
        periodic boundary condition and return the shifted coordinate. If
        `coord` is already in the box, return `coord`.

        :type coord: tuple
        :rtype: tuple
        """
        Dims = len(coord)
        NewCoord = [0,] * Dims
        for i in range(Dims):
            NewCoord[i] = coord[i] % self._GridSize[i]
        return tuple(NewCoord)

    def getBoxShell(self, center, thickness):
        """Given a grid coordinate `center`, return all the grid coordinates that form a
        box shell around `center` with `thickness`, along with the center cell
        itself. For example, in 3D, if `thickness` is 1, return the 3x3x3 cells
        centered at `center`.

        :type center: tuple
        :type thickness: int
        """
        Dims = self.Boundary.Dims
        Delta = [0] * Dims
        Max = thickness * 2

        # We know the total number of coordinates in the result, so just loop
        # that many times.
        for Count in range((thickness * 2 + 1) ** Dims):
            Coord = tuple(center[i] - thickness + Delta[i]
                          for i in range(Dims))
            yield self.senitizeGridCoord(Coord)

            # Calculate the next coordinate.
            for i in range(-1, -Dims - 1, -1):
                Delta[i] += 1
                # Affecting higher digit?
                if Delta[i] > Max:
                    Delta[i] = 0
                else:
                    break

    def findNearest(self, v, n):
        """Given a location `v`, find the `n` nearest neighbors of it, and yield the
        asscociated data with them.

        :type v: Matrix.Vector
        :type n: int
        """
        Thickness = 0
        CenterGrid = self.vec2GridCoord(v)
        while True:
            GridContent = set()
            for Coord in self.getBoxShell(CenterGrid, Thickness):
                GridContent |= self.pointsAndDataAtGrid(Coord)

            Thickness += 1
            if len(GridContent) >= n + 1:
                break

        # We have found a big enough shell to contain n neighbors,
        # however to ensure these are the nearest, we need to
        # investigate the next thicker shell.
        Payloads = set()
        for Coord in self.getBoxShell(CenterGrid, Thickness):
            for Loc, Data in self.pointsAndDataAtGrid(Coord):
                if Loc != v:
                    Payloads.add((Loc, Data))

        for Ele in sorted(Payloads,
                          key=lambda Ele: self.Boundary.dist(v, Ele[0]))[:n]:
            yield Ele[1]

def readPDB(pdb_file):

    with open(pdb_file, 'r') as f:
        Mols = molecule.Molecule.loadFromPDB(f, split=True)

    Center = Mols["PROA"]
    Waters = Mols["SOLV"]

    return (Waters, Center)

def readXYZ(xyz_file):
    with open(xyz_file, 'r') as f:
        Data = molecule.Molecule.loadFromXYZ(f, split=True)

    return Data

def splitBox(mole, bound, center, cutoff, atom_filter=lambda x: True):
    """Split molecule `mole` into 2 parts: The atoms inside a box shell centered at
    `center` with size range `cutoff`, and the ones that are outside. Return 2
    molecules. Distance is calculated with boundary condition `bound`.

    :type mole: molecule.Molecule
    :type bound: molecule.Boundary
    :type center: Matrix.Vector3D
    :type cutoff: tuple
    :rtype: tuple
    """
    MolInside = molecule.Molecule()
    MolOutside = molecule.Molecule()

    for Atom in mole:
        if not atom_filter(Atom):
            continue

        if not bound.inBox(center, Atom.Loc, (cutoff[0],) * 3) and \
           bound.inBox(center, Atom.Loc, (cutoff[1],) * 3):
            MolInside.addAtom(Atom)
        else:
            MolOutside.addAtom(Atom)

    return MolInside, MolOutside

def moleShellCutoff(mol_center, mol_around, cutoff, boundary,
                    atom_filter_center=lambda x: True,
                    atom_filter_around=lambda x: True):
    """Filter atoms in `mol_center` with `atom_filter_center` into set A, and filter
    atoms in `mol_around` with `atom_filter_around` into set B. Return all atoms
    in B as a molecule who are within a distance range of any atoms in A. The
    distance range is specified by the 2-tuple `cutoff`.

    :type mol_center: molecule.Molecule
    :type mol_around: molecule.Molecule
    :type cutoff: tuple
    :rtype: molecule.Molecule
    """

    Center = tuple(at for at in mol_center if atom_filter_center(at))
    Result = molecule.Molecule()
    Ids = set()

    for Atom in mol_around:
        if not atom_filter_around(Atom):
            continue

        for AtomCenter in Center:
            if boundary.inBox(AtomCenter.Loc, Atom.Loc, (cutoff[1],) * 3) and \
               not boundary.inBox(AtomCenter.Loc, Atom.Loc, (cutoff[0],) * 3):
                if id(Atom) not in Ids:
                    Result.addAtom(Atom)
                    Ids.add(id(Atom))
    return Result

def findAllClusters(mole, n, boundary, gridifier=None, atom_filter=lambda x: True):
    """Yield all nearest-n-neighbor clusters in molecule `mole` with boundary
    condition `boundary`. Each yielded cluster is a 2-tuple, whose 1st element
    is the center atom, and the 2nd a tuple of neighbor atoms.
    """
    for Atom in mole:
        if not atom_filter(Atom):
            continue
        if gridifier is None:
            Neighbors = tuple(findNearest(mole, Atom.Loc, n, boundary))
        else:
            Neighbors = tuple(gridifier.findNearest(Atom.Loc, n))
        yield (Atom, Neighbors)

def cosFromTriangle(v1, v2, v3, boundary):
    """Return cos(angle v2-v1-v3)."""

    a = boundary.dist(v1, v2)
    b = boundary.dist(v1, v3)
    c = boundary.dist(v2, v3)

    cos = (a*a + b*b - c*c) / (2.0 * a * b)
    return cos

def orderParam(center, vectors, boundary):
    """Calculate 1 - 3/8 \sum_{j=1}^3 \sum_{k=j+1}^4 ( \cos \theta_{jk} + 1/3)^2.

    :type center: Matrix.Vector3D
    :type vectors: tuple
    :rtype: tuple
    """
    Coss= []
    for j in range(len(vectors) - 1):
        for k in range(j+1, len(vectors)):
            Coss.append(cosFromTriangle(center, vectors[j], vectors[k], boundary))

    Logger.debug("Cosines: " + str(Coss))
    return 1.0 - sum((cos + 0.3333333333333333) ** 2 for cos in Coss) * 0.375

def doFrame(mol, bound, cutoff_type, cutoff, grid_size, center_atom, water_atom,
            legacy_method=False, frame_output=None):
    TimeStart = time.time()

    Bound = bound
    CutoffType = cutoff_type
    Cutoff = cutoff

    Waters = molecule.Molecule()
    Center = molecule.Molecule()

    for Atom in mol:
        if Atom.ID == center_atom:
            Center.addAtom(Atom)
        elif Atom.ID == water_atom:
            Waters.addAtom(Atom)

    if legacy_method:
        Grid = None
    else:
        Grid = Gridifier(Bound, (grid_size,)*3)

    if CutoffType == "shell":
        Inside = moleShellCutoff(Center, Waters, Cutoff, Bound)
    elif CutoffType == "box":
        Inside, _ = splitBox(Waters, Bound, Center.GeoCenter, Cutoff)
    else:
        Inside = Waters

    # if Args.OutputProtein:
    #     with open(Args.OutputProtein, 'w') as f:
    #         Center.saveAsPDB(f)
    # if Args.OutputWater:
    #     with open(Args.OutputWater, 'w') as f:
    #         Inside.saveAsPDB(f)

    Orders = []

    if not legacy_method:
        for Atom in Inside:
            Grid.addPoint(Atom.Loc, Atom)

    for Cluster in findAllClusters(Inside, 4, Bound, Grid):
        Logger.debug("{} -- {}".format(Cluster[0], ' '.join(map(str, Cluster[1]))))
        NewOrder = orderParam(Cluster[0].Loc, tuple(at.Loc for at in Cluster[1]),
                              Bound)
        if frame_output is not None:
            print(NewOrder, file=frame_output)
        Orders.append(NewOrder)

    TimeStop = time.time()

    Time = TimeStop - TimeStart
    print("Frame time: {:.3f}s".format(Time), end='\r', file=sys.stderr)

    return (sum(Orders) / len(Orders), Orders)

def getFileType(file_name):
    """Get the file type of the file in question by taking the lower case of the
    extension name, without the leading dot.
    """
    return os.path.splitext(file_name)[1][1:].lower()

def main():
    import argparse

    Parser = argparse.ArgumentParser(description='Process some integers.')
    Parser.add_argument('PDBPath', metavar='PDB_FILE', type=str,
                        help='The input PDB or XYZ file.')
    Parser.add_argument('-o', "--output", metavar='FILE', type=str, default='-',
                        dest="Output",
                        help="The output file to write order parameters into. "
                        "Default: print to stdout.")
    Parser.add_argument("-t", "--file-type", metavar="TYPE", type=str,
                        choices=("pdb", "xyz"), default=None, dest="FileType",
                        help="Type of input file (pdb or xyz). Default: "
                        "automatically determined by file name.")
    Parser.add_argument("-s", "--box-size", metavar="X", type=float,
                        default=42, dest="BoxSize",
                        help="Box size of the system. Only used for PDB input. "
                        "Default: %(default)s")
    Parser.add_argument("-a", "--print-average", default=False,
                        action="store_true", dest="Avg",
                        help="Print the average order at the end.")
    Parser.add_argument("-j", "--parallel", default=None, dest="Parallel",
                        type=int,
                        help="Number of frames to calculate simultaneously. "
                        "Default: automatically determined by CPU count.")
    Group = Parser.add_mutually_exclusive_group()
    Group.add_argument("-b", "--box-cutoff", metavar=("MIN", "MAX"), type=float,
                       default=None, nargs=2, dest="BoxCutoff",
                       help="Only take water molecules who are within a box "
                       "shell of inner size of MIN and outer size of MAX. "
                       "Cannot be used with -c or -n. Note that "
                       "the sizes are the “diameter” of the box. "
                       "Default: Don't use box cutoff, use shell cutoff.")
    Group.add_argument('-c', "--shell-cutoff", metavar=('MIN', "MAX"), type=float,
                       default=(0, 3.75), nargs=2, dest="ShellCutoff",
                       help="Only take water molecules whose distance to an to any "
                       "CB atoms in the protein (inside) is between MIN and "
                       "MAX. Cannot be used with -b or -n. Note "
                       "that the distance is the “radius” of the shell. "
                       "If none "
                       "of -c, -b or -n is specified, this is the "
                       "default with value %(default)s.")
    Group.add_argument("-n", "--no-cutoff", default=False, action="store_true",
                       dest="NoCutoff",
                       help="Don't cut off water molecules. Cannot be used "
                       "with -b or -c.")
    # Parser.add_argument("-w", "--output-water", metavar="FILE", default=None,
    #                     type=str, dest="OutputWater",
    #                     help="Write the O atoms in the water molecules "
    #                     "selected by either -b or -c to a PDB FILE.")
    # Parser.add_argument("-p", "--output-protein", metavar="FILE", default=None,
    #                     type=str, dest="OutputProtein",
    #                     help="Write the CB atoms in the protein to a PDB FILE.")

    Parser.add_argument("-p", "--center-atom", default="CB", dest="AtomCenter",
                        metavar="ATOM",
                        help="The ID of center atoms of the shell cutoff. "
                        "Default: %(default)s")
    Parser.add_argument("-w", "--water-atom", default="OH2", dest="AtomWater",
                        metavar="ATOM",
                        help="The ID of water atoms. These are the atoms that "
                        "form the tetrahedra. Default: %(default)s")
    Parser.add_argument("-g", "--grid-size", default=20, dest="GridSize",
                        type=int, metavar="N",
                        help="Size of grid used to find nearest neighbors. "
                        "This is the number of cells in each direction, not "
                        "the size of the cell. You should tune this number "
                        "to achieve best performance. Ignored when using "
                        "--legacy. Default: %(default)s")
    Parser.add_argument("--legacy", default=False, action="store_true",
                        dest="LegacyMethod",
                        help="Use the dumb method to find nearest neighbors. "
                        "This could be preferrable if the number of water "
                        "molecules in each frame is small (like several dozen).")
    Parser.add_argument("--verbose", default=False, action="store_true",
                        dest="Verbose",
                        help="Print debug messages.")

    Args = Parser.parse_args()

    if Args.Verbose:
        Logger.setLevel(logging.DEBUG)

    PDBPath = Args.PDBPath
    BoxSize = Args.BoxSize
    GridSize = Args.GridSize

    if Args.NoCutoff is True:
        Logger.info("No cutoff.")
        Cutoff = None
        CutoffType = None
    elif Args.BoxCutoff is not None:
        Logger.info("Applying box cutoff...")
        CutoffType = "box"
        Cutoff = Args.BoxCutoff
    else:
        Logger.info("Applying shell cutoff...")
        CutoffType = "shell"
        Cutoff = Args.ShellCutoff

    FileType = getFileType(Args.PDBPath)
    if Args.FileType is not None:
        FileType = Args.FileType

    if Args.Output == '-':
        OutputFile = sys.stdout
    elif Args.Output is None:
        OutputFile = None
    else:
        OutputFile = open(Args.Output, 'w')

    if FileType == "pdb":
        with open(Args.PDBPath, 'r') as f:
            Mol = molecule.Molecule.loadFromPDB(f)

        Bound = molecule.Boundary().setWalls(
            Matrix.Vector3D(-BoxSize/2.0, -BoxSize/2.0, -BoxSize/2.0),
            Matrix.Vector3D(BoxSize/2.0, BoxSize/2.0, BoxSize/2.0))

        Avg, _ = doFrame(Mol, Bound, CutoffType, Cutoff, GridSize, Args.AtomCenter,
                         Args.AtomWater, Args.LegacyMethod, OutputFile)

        if Args.Avg:
            print(Avg)

    elif FileType == "xyz":
        with open(Args.PDBPath, 'r') as f:
            Mols = molecule.Molecule.loadFromXYZ(f)

        # In order to properly handle C-c, the slaves must not handle it.
        Pool = Mp.pool.Pool(Args.Parallel,
                            lambda: signal.signal(signal.SIGINT, signal.SIG_IGN))
        FrameValues = []
        Results = []

        for Data in Mols:
            Bound = Data["box"]
            Mol = Data["mol"]
            Results.append(
                Pool.apply_async(doFrame, (Mol, Bound, CutoffType, Cutoff, GridSize,
                                           Args.AtomCenter, Args.AtomWater,
                                           Args.LegacyMethod, None)))
        Pool.close()
        try:
            Pool.join()
            for Result in Results:
                FrameValues += Result.get()[1]
        except KeyboardInterrupt:
            Logger.info("Killing slaves...")
            Pool.terminate()
            return 2

        if Args.Output == '-':
            OutputFile = sys.stdout
        else:
            OutputFile = open(Args.Output, 'w')

        for Value in FrameValues:
            print(Value, file=OutputFile)

        if Args.Output != '-':
            OutputFile.close()

        if Args.Avg:
            print(sum(FrameValues) / float(len(FrameValues)))

    else:
        Logger.fatal("Unknown file type: " + FileType)
        return 1

    if Args.Output != '-':
        OutputFile.close()

    return 0

if __name__ == "__main__":
    sys.exit(main())
