data = readmatrix('output_loc_1.xlsx');
%disp(data);
x = zeros(224, 301);
y = zeros(224, 301);
pi = acos(-1);
theta = 0:0.1:50 * pi;
alpha = 0.55 / (2 * pi);
r = theta .* alpha;
[rx, ry] = polar2cartesian(r, theta);

for i = 1:size(data, 2)

    for j = 1:size(data, 1) / 2
        x(j, i) = data(j * 2, i);
        y(j, i) = data(j * 2 + 1, i);
    end

end

plot(rx, ry, color = 'black');
hold on;

for i = 1:size(data, 2)
    p = plot(x(:, i), y(:, i), 'g');
    s = scatter(x(:, i), y(:, i), 'r');
    disp(i - 1);
    drawnow;
    pause;
    delete(p);
    delete(s);
end

function [x, y] = polar2cartesian(r, theta)
    x = zeros(size(r, 2));
    y = zeros(size(r, 2));

    for i = 1:size(r, 2)
        x(i) = r(i) * cos(theta(i));
        y(i) = r(i) * sin(theta(i));
    end

end
