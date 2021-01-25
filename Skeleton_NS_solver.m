%%
clear all
close all
clc

% The system that you need to solve will be singular. Matlab gives you a
% warning at each time step. To switch of this warning, remove the comment
% in the next line

warning on

% 00D#MMXX#

% This file contains the skeleton of the program which will solve the lid
% driven cavity problem on a unit square. The parts that have to be
% supplemented are described in the assignment.
%
% The pieces that need to be filled in are indicated
%

%
% When running the code, determine a suitable time step. A too small time
% step will make the calculation very long, while for a too large time step
% the solution will blow up due to numerical instability.
%

Re = 1000;              % Reynolds number
N = 32;                 % Number of volumes in the x- and y-direction
Delta = 1/N;            % uniform spacing to be used in the mapping to compute tx

filename = "results_N_"+N+".mat"; %filename to save workspace to for post-processing

% Determine a suitable time step and stopping criterion, tol

tol =1e-6;             % tol determines when steady state is reached and the program terminates

% wall velocities
U_wall_top = -1;
U_wall_bot = 0;
U_wall_left = 0;
U_wall_right = 0;
V_wall_top = 0;
V_wall_bot = 0;
V_wall_left = 0;
V_wall_right = 0;

%
%   Generation of a non-uniform mesh
%

%
%   tx are the coordinates of the nodal points on the outer-oriented 
%   primal mesh
%

tx = zeros(1,N+1);
for i=1:N+1
    xi = (i-1)*Delta;
    tx(i) = 0.5*(1. - cos(pi*xi));
end

% Mesh width on the outer-oriented primal mesh
th = zeros(N,1);
th = tx(2:N+1) - tx(1:N);

%
%  x are the coordinates of the nodal points on the dual inner-orineted 
%  mesh (including endpoints 0 and 1)
%  h contains the edge lengths on the inner-oriented dual mesh
%
x = 0.5*(tx(1:N) + tx(2:N+1));
x = [0 x 1];

h = zeros(N+1,1);
h = x(2:N+2) - x(1:N+1);

% Stable time step. Note that this is a conservative estimate, so it is
% possible to increase this time step somewhat.

dt = min(min(h),0.5*Re*min(h)^2);
%
%   Initial condition u=v=0
%
%   Both u and v will be stored in one big vector called 'u'
%
%   The vector u only contains the true unknowns, not the velocities that
%   are prescribed by the boundary conditions
%
%   The vector u contains the *inner-oriented* circulations as unknowns

u = zeros(2*N*(N+1),1);

% Set up the Incindence matrix 'tE21' which connects the fluxes to the
% volumes. Use the orientation described in the assignment.

%
%  Inserting boundary conditions for normal velocity components and store
%  this part in the vector u_norm, see assignemnt.
%

%  Set up the sparse, outer-oriented incidence matrix tE10.



%  Set up the sparse, inner-oriented  incidence matrix E10


%  Set up the (extended) sparse, inner-oriented incidence matrix E21


%  Split off the prescribed tangential velocity and store this in 
%  the vector u_pres


%  Set up the Hodge matrices Ht11 and H1t1


%  Set up the Hodge matrix Ht02



%
% The prescribed velocties will play a role in the momentum equation
%
u_pres_tvort=Ht02*u_pres; %U_pres to outer oriented 0 form representing contribution of boundary conditions to point wise vorticity
u_pres = H1t1*E21'*Ht02*u_pres; %U_pres to inner oriented 1 forms


% Now all matrices are set up and the time stepping can start. 'iter' will
% record the number of time steps. This allows you to give output after a
% preselected number of time steps.
%
% 'diff' will be the maximal du/dt or dv/dt. If 'diff' is sufficiently
% small, steady state has been reached. Determine a suitable value for
% 'tol'
%

convective = zeros(2*N*(N+1),1);
ux_xi = zeros((N+1)*(N+1),1);
uy_xi = zeros((N+1)*(N+1),1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%    From here the code enters the time-stepping loop and no new parts in
%    the code need to be inserted. If done correctly, everything should
%    work now.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

diff = 1;
iter = 1;


% Set up the matrix for the Poisson equation    

A = -tE21*Ht11*tE21';

% Perform an LU-decomposition for the pressure matrix A

[L,U] = lu(A);

% Abbreviation for some matrix products which are constant in the time loop

VLaplace = H1t1*E21'*Ht02*E21;
DIV = tE21*Ht11;

while diff > tol
%while iter < 10
        
    %Vector xi is obtained. It corresponds with the point-wise vorticity
    %at each cell
    
    %Vectors ux_xi and uy_xi correspond with the multiplications of
    %xi with the horizontal and vertical velocity components at each cell.
    %Only the cells required for vector convective are calculated. The
    %ordering of each vector with respect to the ordering of cells in the
    %grid is different (left to right for ux_xi and bottom to top for
    %uy_xi)
    
    xi = Ht02@E21*u + u_pres_vort;
    
    for i=1:N+1
        for j=1:N+1
            k = j + (i-1)*(N+1); 
            if j==1
                ux_xi(k) = U_wall_bot*xi(i+(j-1)*(N+1));    
                uy_xi(k) = V_wall_left*xi(j+(i-1)*(N+1));
            elseif j==N+1
                ux_xi(k) = U_wall_top*xi(i+(j-1)*(N+1));
                uy_xi(k) = V_wall_right*xi(j+(i-1)*(N+1));
            else
                ux_xi(k) = (u(i+(j-1)*(N+1))+u(i+(j-2)*(N+1)))*xi(i+(j-1)*(N+1))/(2.*h(i));                      
                uy_xi(k) = (u(N*(N+1)+j+(i-1)*N) + u(N*(N+1)+j-1+(i-1)*N))*xi(j+(i-1)*(N+1))/(2.*h(i));  
            end
        end
    end

    for  i=1:N
        for j=1:N+1
            convective(j+(i-1)*(N+1)) = -(uy_xi(j+(i-1)*(N+1))+uy_xi(j+i*(N+1)))*h(j)/2.;
            convective(N*(N+1)+i+(j-1)*N) = (ux_xi(j+(i-1)*(N+1))+ux_xi(j+i*(N+1)))*h(j)/2.;
        end
    end
    

    % Set up the right hand side for the Poisson equation for the pressure
    
    rhs_Poisson  =   DIV*(u/dt  - convective - VLaplace*u/Re - u_pres/Re) + u_norm/dt; 
    
    % Solve for the new pressure
    
    temp = L\rhs_Poisson;
    p = U\temp;
    
    % Store the velocity from the previous time step in the vector u_old
    
    uold = u;
    
    % Udate the velocity field
    
    u = u - dt* (convective - tE21'*p + VLaplace*u/Re + u_pres/Re); 
    
    %
    %  Every other 1000 iterations check whether you approach steady state
    %  and check whether yopu satisfy conservation of mass. The largest
    %  rate at which mass is destroyed or created is denoted by 'maxdiv'.
    %  This number should be very small, in the order of machine precision.
    
    if mod(iter,1000) == 0
    
        maxdiv = max(DIV*u + u_norm) 
        
        diff = max(abs(u-uold))/dt
        
    end
    iter = iter + 1;
end



