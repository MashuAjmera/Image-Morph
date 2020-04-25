import numpy as np
import cv2
import sys


# creating class for pixels on image (x and y coordinates)
class pts:
	def __init__(self, x,y):
		self.x=x
		self.y=y


# creating class for triangle (3 points)
class triangle:
	def __init__(self, vertex):
		self.vertex=vertex


# creating class for image (contains image, control points and triangulation)
class data:
	def __init__(self, img, points, tri ):
		self.img = img
		self.points = points
		self.tri = tri


class Delaunay:
  # Class to compute a Delaunay triangulation
  def __init__(self, center=(0, 0), radius=9999):
    """center -- Optional position for the center of the frame. Default (0,0)
    radius -- Optional distance from corners to the center.
    """
    center = np.asarray(center)
    # Create coordinates for the corners of the frame
    self.coords = [center+radius*np.array((-1, -1)),
                    center+radius*np.array((+1, -1)),
                    center+radius*np.array((+1, +1)),
                    center+radius*np.array((-1, +1))]

    # Create two dicts to store triangle neighbours and circumcircles.
    self.triangles = {}
    self.circles = {}

    # Create two CCW triangles for the frame
    T1 = (0, 1, 3)
    T2 = (2, 3, 1)
    self.triangles[T1] = [T2, None, None]
    self.triangles[T2] = [T1, None, None]

    # Compute circumcenters and circumradius for each triangle
    for t in self.triangles:
      self.circles[t] = self.circumcenter(t)

  def circumcenter(self, tri):
    # Compute circumcenter and circumradius of a triangle.
    pts = np.asarray([self.coords[v] for v in tri])
    pts2 = np.dot(pts, pts.T)
    A = np.bmat([[2 * pts2, [[1],
                              [1],
                              [1]]],
                  [[[1, 1, 1, 0]]]])

    b = np.hstack((np.sum(pts * pts, axis=1), [1]))
    x = np.linalg.solve(A, b)
    bary_coords = x[:-1]
    center = np.dot(bary_coords, pts)

    # radius = np.linalg.norm(pts[0] - center) # euclidean distance
    radius = np.sum(np.square(pts[0] - center))  # squared distance
    return (center, radius)

  def inCircle(self, tri, p):
    # Check if point p is inside of precomputed circumcircle of tri.
    center, radius = self.circles[tri]
    return np.sum(np.square(center - p)) <= radius

  def addControlPoint(self, p):
    # Add a point to the current DT, and refine it using Bowyer-Watson.
    p = np.asarray(p)
    idx = len(self.coords)
    # print("coords[", idx,"] ->",p)
    self.coords.append(p)

    # Search the triangle(s) whose circumcircle contains p
    bad_triangles = []
    for T in self.triangles:
      if self.inCircle(T, p):
        bad_triangles.append(T)

    # Find the CCW boundary (star shape) of the bad triangles,
    # expressed as a list of edges (point pairs) and the opposite
    # triangle to each edge.
    boundary = []
    # Choose a "random" triangle and edge
    T = bad_triangles[0]
    edge = 0
    # get the opposite triangle of this edge
    while True:
      # Check if edge of triangle T is on the boundary...
      # if opposite triangle of this edge is external to the list
      tri_op = self.triangles[T][edge]
      if tri_op not in bad_triangles:
        # Insert edge and external triangle into boundary list
        boundary.append((T[(edge+1) % 3], T[(edge-1) % 3], tri_op))

        # Move to next CCW edge in this triangle
        edge = (edge + 1) % 3

        # Check if boundary is a closed loop
        if boundary[0][0] == boundary[-1][1]:
          break
      else:
        # Move to next CCW edge in opposite triangle
        edge = (self.triangles[tri_op].index(T) + 1) % 3
        T = tri_op

    # Remove triangles too near of point p of our solution
    for T in bad_triangles:
      del self.triangles[T]
      del self.circles[T]

    # Retriangle the hole left by bad_triangles
    new_triangles = []
    for (e0, e1, tri_op) in boundary:
      # Create a new triangle using point p and edge extremes
      T = (idx, e0, e1)

      # Store circumcenter and circumradius of the triangle
      self.circles[T] = self.circumcenter(T)

      # Set opposite triangle of the edge as neighbour of T
      self.triangles[T] = [tri_op, None, None]

      # Try to set T as neighbour of the opposite triangle
      if tri_op:
        # search the neighbour of tri_op that use edge (e1, e0)
        for i, neigh in enumerate(self.triangles[tri_op]):
          if neigh:
            if e1 in neigh and e0 in neigh:
              # change link to use our new triangle
              self.triangles[tri_op][i] = T

      # Add triangle to a temporal list
      new_triangles.append(T)

    # Link the new triangles each another
    N = len(new_triangles)
    for i, T in enumerate(new_triangles):
      self.triangles[T][1] = new_triangles[(i+1) % N]   # next
      self.triangles[T][2] = new_triangles[(i-1) % N]   # previous

  def exportTriangles(self):
    # Export the current list of Delaunay triangles
    # Filter out triangles with any vertex in the extended BBox
    return [(a-4, b-4, c-4)
            for (a, b, c) in self.triangles if a > 3 and b > 3 and c > 3]


# taking input from user for control points in images (only 3 points)
def mouseHandler(event,x,y,flags,param):
  if event==cv2.EVENT_LBUTTONDOWN:
    imgdata = param
    if len(imgdata.points)<100:
      
      #creating a dot for user accessibility (circle radius-3 color-yellow)
      cv2.circle(imgdata.img,(x,y), 3, (0,255,255), 5)
      cv2.imshow("Image",imgdata.img)
      
      #displaying chosen coordinates
      print(x," ",y)		
      
      #storing chosen points
      dot=pts(y,x)		
      imgdata.points.append(dot)


# function to make a triangle if 3 vertices are given
def makeTriangle(a,  b, c):
	temp=triangle([])
	temp.vertex.append(a)
	temp.vertex.append(b)
	temp.vertex.append(c)
	return temp


# sorting function comparator based on y coordinates of two points
def compareY(val):
    return val.y


# function to create triangulation of an image given points on 2d plane (for 3 points only(not using any algorithm))
def triangulation(points, rows, cols):
  temp=[]
  seeds=[]
  points.sort(key=compareY)

  # Boundary points (Top-Left,Bottom-Left,Bottom-Right,Top-Right)
  seeds.append(pts(0,0))
  seeds.append(pts(rows-1,0))
  seeds.append(pts(rows-1,cols-1))
  seeds.append(pts(0,cols-1))
  for p in points:
    seeds.append(p)
  
  # Delaunay Triangulation based on control points
  dt = Delaunay()
  for s in seeds:
    dt.addControlPoint([s.x,s.y])
  for t in dt.exportTriangles():
    temp.append(makeTriangle(seeds[t[0]],seeds[t[1]],seeds[t[2]]))
	
  return temp


def exportDelaunayImage(img, triangles,lineColours):
      for t in triangles :
    # Triangle Coordinates
    p1 = [t.vertex[0].x,t.vertex[0].y]
    p2 = [t.vertex[1].x,t.vertex[1].y]
    p3 = [t.vertex[2].x,t.vertex[2].y]

    # Drawing triangles
    cv2.line(img, (p1[1],p1[0]), (p2[1],p2[0]), lineColours, 1)
    cv2.line(img, (p2[1],p2[0]), (p3[1],p3[0]), lineColours, 1)
    cv2.line(img, (p3[1],p3[0]), (p1[1],p1[0]), lineColours, 1)

  cv2.imshow("Image",img)
  return img


# function to define points in intermediate images (using weighted average)
def fillmorph(p3,p1,p2,alpha):
  i=0
  while i<len(p1):
    xmorphed= (1-alpha)*p1[i].x + alpha*p2[i].x
    ymorphed= (1-alpha)*p1[i].y + alpha*p2[i].y

    dot= pts(xmorphed,ymorphed)
    p3.append(dot)
    i+=1


# creating affine matrix for transformation given 3 points (this function returns inverse of the actual matrix for reverse mapping)
def getAffine(tri1,tri2):
	# matrix based on triangle1 vertices (3 Points)
	A=np.array([
			[tri1.vertex[0].x,tri1.vertex[0].y,1,0,0,0],
			[0,0,0,tri1.vertex[0].x,tri1.vertex[0].y,1],
			[tri1.vertex[1].x,tri1.vertex[1].y,1,0,0,0],
			[0,0,0,tri1.vertex[1].x,tri1.vertex[1].y,1],
			[tri1.vertex[2].x,tri1.vertex[2].y,1,0,0,0],
			[0,0,0,tri1.vertex[2].x,tri1.vertex[2].y,1]
        ])
		
	# matrix based on triangle2 vertices
	B=np.array([tri2.vertex[0].x,tri2.vertex[0].y,tri2.vertex[1].x,tri2.vertex[1].y,tri2.vertex[2].x,tri2.vertex[2].y])
	
	# variables in transformation matrix
	H = np.dot(np.linalg.inv(A), B)
	
	C=np.array([
			[H[0],H[1],H[2]],
			[H[3],H[4],H[5]],
			[0,0,1]
		])
		
	# returning inverse of transformation matrix for reverse mapping
	return np.linalg.inv(C)


# getting orientation of 3 points (ACW or CW)
def orientation( x1,  y1,  x2, y2, px,  py):
	o= ((x2-x1)*(py-y1))-((px-x1)*(y2-y1))
	return 1 if o>0 else -1 if o<0 else 0


# finding if a point lies inside or outside a triangle (using orientation)
def isinTri(t,  p):
	o1= orientation(t.vertex[0].x,t.vertex[0].y,t.vertex[1].x,t.vertex[1].y,p.x,p.y)
	o2= orientation(t.vertex[1].x,t.vertex[1].y,t.vertex[2].x,t.vertex[2].y,p.x,p.y)
	o3= orientation(t.vertex[2].x,t.vertex[2].y,t.vertex[0].x,t.vertex[0].y,p.x,p.y)
	
	return (o1==o2 and o2==o3)


# finding suitable triangle for a point
def findTriangle(t,  p):
	for i in range(len(t)):
		if(isinTri(t[i],p)):
			return i
	return -1


# creating intermediate images based on different values of alpha
def morphedImage(alpha,srcdata,destdata):
  morphdata=data([], [], [])
		
  # initialising intermediate image
  morphdata.img=np.empty([destdata.img.shape[0], destdata.img.shape[1], destdata.img.shape[2]]) 
		
  # control points based on weighted average
  fillmorph(morphdata.points,srcdata.points,destdata.points,alpha)
		
  # triangulating above computed points
  morphdata.tri=triangulation(morphdata.points,morphdata.img.shape[0],morphdata.img.shape[1])
    	
  srcaffine = []
  destaffine = []
    	
  # getting transformation matrix for each corresponding triangle
  i=0
  while(i<len(morphdata.tri)):
    srcaffine.append(getAffine(srcdata.tri[i],morphdata.tri[i]))
    destaffine.append(getAffine(destdata.tri[i],morphdata.tri[i]))
    i+=1
  i=0    
  while(i<morphdata.img.shape[0]):
    j=0
    while(j<morphdata.img.shape[1]):
      dot=pts(i,j)
			# finding triangle for point
      k= findTriangle(morphdata.tri,dot)
				
      # if no triangle found skip the point
      if(k==-1):
        j+=1
        continue
				
      # finding corresponding points in source and destination image (reverse mapping)
      orig=[i,j,1]
      modified= orig
      srcpoints=np.dot(srcaffine[k],modified)
      destpoints=np.dot(destaffine[k],modified)
    			
      # typecasting double points to int
      srcx= int(np.round(srcpoints[0]))
      srcy= int(np.round(srcpoints[1]))
      destx= int(np.round(destpoints[0]))
      desty= int(np.round(destpoints[1]))

      # using color interpolation to assign color values for each pixel
      morphdata.img[i,j]= (1-alpha)*srcdata.img[srcx,srcy] + alpha*destdata.img[destx,desty]
      j+=1

    i+=1
    	
  return morphdata.img


if __name__ == '__main__' :
  # reading source image and checking for errors
  filename1 = input("Enter source filename: ")    
  src = cv2.imread(filename1)
  srcdata=data(src.copy() , [], [])
	
  # creating window
  cv2.namedWindow("Image")
	
  # user input from mouse for source image
  print("Kindly click on your choice of control points:")	
  cv2.setMouseCallback("Image",mouseHandler,srcdata)
  cv2.imshow("Image",srcdata.img)
  cv2.waitKey(0)

  # triangulating source image
  srcdata.tri=triangulation(srcdata.points,srcdata.img.shape[0],srcdata.img.shape[1])
  srctri=exportDelaunayImage(src.copy(),srcdata.tri,(255,0,0))
  cv2.waitKey(0)
  cv2.imwrite("src_triangulate.jpg",srctri)

  # retrieving original image to undo any modification (dots on images) 
  srcdata.img=src

  #reading destination image and checking for errors
  filename2= input("Enter destination filename: ")
  dest =cv2.imread(filename2)
  destdata=data(dest.copy() , [], [])

  # user input from mouse for destination image
  print("Kindly click on your choice of control points:")	
  cv2.setMouseCallback("Image",mouseHandler,destdata)
  cv2.imshow("Image",destdata.img)
  cv2.waitKey(0)

  # triangulating destination image
  destdata.tri=triangulation(destdata.points,destdata.img.shape[0],destdata.img.shape[1])
  desttri=exportDelaunayImage(dest.copy(),destdata.tri,(0,255,255))	
  cv2.waitKey(0)
  cv2.imwrite("dest_triangulate.jpg",desttri)

  # retrieving original image to undo any modification (dots on images) 
  destdata.img=dest

  # closing any windows created
  cv2.destroyAllWindows()
	
  # creating and saving intermediate frames
  step = float(input("Enter 1/number of intermediate frames: "))
  alpha=0
  while(alpha<1):
    alpha+=step
    name= "morphed_" + str(step) + ".jpg"
    cv2.imwrite(name,morphedImage(alpha,srcdata,destdata))
    print("Saved intermediate frame",name)