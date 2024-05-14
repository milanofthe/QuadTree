###################################################################################
##
##                          QuadTree definition
##
##                          Milan Rother 2023/24
##
###################################################################################


# IMPORTS =========================================================================

import numpy as np

from quadtree.cell import Cell



# QuadTree CLASS ==================================================================

class QuadTree:
    """
    class for managing the 2D Quadtree datastructure

    INPUTS : 
        bounding_box : (list of tuples) bounding box of the quadtree root cell
        n_initial_x  : (int) number of initial root cell splits along x-axis
        n_initial_y  : (int) number of initial root cell splits along y-axis
    """

    def __init__(self, bounding_box=[(-1, -1), (1, 1)], n_initial_x=2, n_initial_y=2):

        self.bounding_box = bounding_box

        #determine root cell size and center from  bounding box
        bb_x, bb_y = zip(*self.bounding_box)
        center = sum(bb_x)/2, sum(bb_y)/2
        size = abs(min(bb_x) - max(bb_x)), abs(min(bb_y) - max(bb_y))

        #initialize quadtree root cell
        self.root = Cell(center, size)

        #initial root cell splits
        self.root.split(n_initial_x, n_initial_y)


    def __len__(self):
        return len(self.get_leafs())


    def get_leafs(self):
        return self.root.get_leafs()


    def get_closest_leaf(self, point):
        return self.root.get_closest_leaf(point)


    def balance(self):
        """
        Ballance all leaf cells of the quadtree by splitting the 
        cells that have more then 2 neighbors in some direction 
        (sometimes this is also called a graded quadtree).
        """

        #balancing flag
        needs_balancing = True

        #balance individual cells until all leafs are balanced
        while needs_balancing:

            needs_balancing = False

            #get the leafs
            leafs = self.get_leafs()

            #iterate all relevant leaf cells
            for cell in leafs:

                if not cell.is_balanced():
                    cell.split(2, 2)
                    needs_balancing = True


    def refine_boundary(self,
                        x_min=False,
                        x_max=False,
                        y_min=False,
                        y_max=False,
                        min_size=0.0):
        """
        Automatically refine leaf cells based on the selected mode

        INPUTS : 
            x_min       : (bool) quadtree refinement at left boundary
            x_max       : (bool) quadtree refinement at right boundary
            y_min       : (bool) quadtree refinement at bottom boundary
            y_max       : (bool) quadtree refinement at top boundary
            min_size    : (float) sets smallest allowed cell size
        """

        for cell in self.get_leafs_at_boundary():

            if ((x_min and cell.is_boundary_W()) or 
                (x_max and cell.is_boundary_E()) or 
                (y_min and cell.is_boundary_S()) or 
                (y_max and cell.is_boundary_N())):
                cell.split(2, 2)


    def refine_edge(self,
                    segments,
                    min_size=0.0,
                    tol=1e-12):
        """
        Automatically refine leaf cells based on the selected mode and geometry. 
        The geometry is provided in the format of line segments that consist of 
        two points (x-y-coords) each. The method checks for all leaf cells if they 
        are intersected by the segments.

        INPUTS : 
            segments : (list of list tuples) set of line segments made of two points each that form path
            min_size : (float) sets smallest allowed cell size
            tol      : (float) numerical tolerance for checking if a cell is cut by the segment
        """ 

        for cell in self.get_leafs_cut_by_segments(segments, tol):
            if min(cell.size) > min_size:
                cell.split(2, 2)


    def get_leafs_cut_by_segments(self, segments, tol=1e-12):
        """
        Retrieve all the leaf cells that are cut by the line segments with some tolerance.

        INPUTS : 
            segments : (list of lists of tuples of floats) list of line segments that are defined by two points each
            tol      : (float) numerical tolerance for checking if a cell is cut by the segment
        """

        relevant_leafs = []
        for cell in self.get_leafs():
            for p1, p2 in segments:
                if cell.is_cut_by_line(p1, p2, tol):
                    relevant_leafs.append(cell)
                    break
        return relevant_leafs


    def get_leafs_at_boundary(self):
        """
        Retrieve the leaf cells at the quadtree boundary.
        """
        return [cell for cell in self.get_leafs() if cell.is_boundary()]


    def get_leafs_inside_polygon(self, polygon):
        """
        Retrieve the leaf cells that are within a polygon

        INPUTS : 
            polygon : (list of tuples of floats) non closed path that defines the polygon
        """
        return [cell for cell in self.get_leafs() if cell.is_inside_polygon(polygon)]


    def get_leafs_outside_polygon(self, polygon):
        """
        Retrieve the leaf cells that are outside of a polygon

        INPUTS : 
            polygon : (list of tuples of floats) non closed path that defines the polygon
        """
        return [cell for cell in self.get_leafs() if not cell.is_inside_polygon(polygon)]


    
